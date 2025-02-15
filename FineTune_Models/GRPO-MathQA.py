#!/usr/bin/env python3
import os
import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    get_scheduler
)
from peft import LoraConfig, TaskType, get_peft_model

from trl import PPOTrainer, PPOConfig
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


BASE_MODEL_PATH = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
ADAPTER_CHECKPOINT = "/home/n/Token-Book/deepseek_fine/checkpoints/qwen2.5-math.1.5b/"
TRAIN_JSON_PATH = "/home/n/Token-Book/eval/Benchmark/MathQA/train_low_250.json"
OUTPUT_DIR = "checkpoints/qwen2.5-GRPO-GOOD"


class ChainOfThoughtDataset(Dataset):
    def __init__(self, tokenizer, json_file, block_size=256):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)  # Assumes data is a list of dicts.
        self.examples = []
        for entry in data:
            prompt = (
                f"Problem: {entry['Problem'].strip()}\n"
                f"Options: {entry['options'].strip()}\n"
                "Please explain your reasoning step-by-step and then provide your final answer.\n"
            )
            rationale = entry.get("Rationale", "").strip()
            if rationale.startswith('"') and rationale.endswith('"'):
                rationale = rationale[1:-1].strip()
            target = f"Chain-of-Thought: {rationale}\nAnswer: {entry['correct'].strip()}\n"
            full_text = prompt + target
            self.examples.append(full_text)
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Ensure the pad_token_id is set to eos_token_id
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.block_size,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = item["input_ids"].clone()

        # Ensure attention mask is present
        if 'attention_mask' not in item:
            item['attention_mask'] = torch.ones_like(item['input_ids'])

        return item


def extract_final_answer(response_text):
    # Extracts the final answer from LLM, accuracy is high 
    response_clean = re.sub(r'\s+', ' ', response_text.strip())
    response_clean = re.sub(r'\*', '', response_clean)

    if not re.search(r'[a-zA-Z0-9]', response_clean):
        return "empty", response_clean[-80:]

    last_chars = response_clean[-80:]
    answer_found = False
    final_answer_extracted = ""

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", last_chars)
    if boxed_match:
        boxed_content = boxed_match.group(1).strip()
        letter_match = re.search(r'([a-eA-E])', boxed_content)
        if letter_match:
            final_answer_extracted = letter_match.group(1).lower()
            answer_found = True
        else:
            try:
                numeric_str_boxed = re.sub(r'\s+', '', boxed_content)
                boxed_number = float(numeric_str_boxed)
                final_answer_extracted = str(boxed_number)
                answer_found = True
            except Exception:
                final_answer_extracted = "none"

    if not answer_found:
        pattern = re.compile(r'(?i)(?:\*{0,2}answer|\*{0,2}solution\s*|answer\s*is).*?([a-e])')
        match = pattern.search(last_chars)
        if match:
            final_answer_extracted = match.group(1).lower()
            answer_found = True

    if not answer_found:
        letter_with_paren = re.search(r'([a-eA-E])\s*\)', last_chars)
        if letter_with_paren:
            final_answer_extracted = letter_with_paren.group(1).lower()
            answer_found = True

    if not answer_found:
        final_answer_extracted = "none"

    return final_answer_extracted, last_chars

def extract_ground_truth(query_text):
    # Extracts the answer from the MathQA data, test.json I think 
    match = re.search(r'Answer:\s*([a-zA-Z0-9]+)', query_text)
    if match:
        return match.group(1).strip().lower()
    return None

def compute_rewards(queries, responses):
    rewards = []
    empty_count = 0
    details = []

    for query, response in zip(queries, responses):
        ground_truth = extract_ground_truth(query)
        extracted, last_chars = extract_final_answer(response)

        if extracted == "empty":
            empty_count += 1
            reward = 0.0
        else:
            if ground_truth is not None and ground_truth.isalpha():
                reward = 1.0 if extracted == ground_truth else 0.0
            else:
                try:
                    extracted_val = float(extracted)
                    truth_val = float(ground_truth)
                    reward = 1.0 if abs(extracted_val - truth_val) < 1e-6 else 0.0
                except Exception:
                    reward = 0.0

        rewards.append(reward)
        details.append({
            "extracted": extracted,
            "ground_truth": ground_truth,
            "last_chars": last_chars,
            "reward": reward
        })

    return rewards, empty_count, details

def run_rl_training(model, tokenizer, ref_model, num_rl_epochs=1):
    ppo_config = PPOConfig(
        model_name=BASE_MODEL_PATH,
        learning_rate=5e-5,
        batch_size=1,
        ppo_epochs=4,
        log_with="tensorboard",
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "tensorboard_logs"))

    dataset = ChainOfThoughtDataset(tokenizer, TRAIN_JSON_PATH, block_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("Starting RL (PPO) fine-tuning...")
    global_step = 0
    for epoch in range(num_rl_epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            queries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            
            responses = []
            for query in queries:
                input_ids_query = tokenizer.encode(query, return_tensors="pt").to(model.device)
                response_ids = model.generate(
                    input_ids=input_ids_query,
                    max_new_tokens=256,  
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=3
                )
                response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
                responses.append(response_text)
            
            rewards, empty_count, details = compute_rewards(queries, responses)
            correct_count = sum(1 for d in details if d["reward"] == 1.0)
            accuracy = correct_count / len(details) if details else 0.0

            writer.add_scalar("ppo/accuracy", accuracy, global_step)

            ppo_stats = ppo_trainer.step(queries, responses, rewards)
            if isinstance(ppo_stats, dict) and "loss" in ppo_stats:
                writer.add_scalar("ppo/loss", ppo_stats["loss"], global_step)

            global_step += 1

    writer.close()


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(base_model, lora_config)

    if os.path.exists(ADAPTER_CHECKPOINT):
        adapter_weights_path = os.path.join(ADAPTER_CHECKPOINT, "pytorch_model.bin")
        if os.path.exists(adapter_weights_path):
            print("Loading adapter weights from", adapter_weights_path)
            adapter_weights = torch.load(adapter_weights_path, map_location="cpu")
            model.load_state_dict(adapter_weights, strict=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128,
        optim="adamw_torch", # I used this one because in the hacky version for arch linux the other ones doesn't work unfortunately.
        evaluation_strategy="no",
        save_steps=200,
        logging_steps=10,
        learning_rate=5e-4,
        warmup_steps=20,
        fp16=True,
        seed=90134,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
    )

    train_dataset = ChainOfThoughtDataset(tokenizer, TRAIN_JSON_PATH)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("Starting supervised training...")
    trainer.train()
    print("Supervised training complete.")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    run_rl_training(model, tokenizer, ref_model=base_model)

if __name__ == "__main__":
    main()

