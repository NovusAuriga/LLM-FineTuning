#!/usr/bin/python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import re

torch.cuda.set_per_process_memory_fraction(0.8, device=0)

# Set the paths
BASE_MODEL_PATH = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
OUTPUT_RL = "checkpoints/qwen2.5-grpo-0.5b"
LOCAL_DATA_PATH = "/home/n/Token-Book/deepseek_fine/datasets/gsm8k/processed/train_data.jsonl"
ADAPTER_CHECKPOINT = "checkpoints/qwen2.5-lora-0.5b"

# Load the local dataset
dataset = Dataset.from_json(LOCAL_DATA_PATH)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Load the LoRA model (pre-trained with LoRA)
lora_config = LoraConfig(
    r=2,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Load base model and apply LoRA
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
base_model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency
model = get_peft_model(base_model, lora_config)

# Load LoRA weights if they exist
if os.path.exists(ADAPTER_CHECKPOINT):
    adapter_weights_path = os.path.join(ADAPTER_CHECKPOINT, "pytorch_model.bin")
    if os.path.exists(adapter_weights_path):
        print(f"Loading LoRA adapter weights from {adapter_weights_path}")
        adapter_weights = torch.load(adapter_weights_path, map_location="cpu")
        model.load_state_dict(adapter_weights, strict=False)

# Reward function to compare the predicted and actual answers
def extract_final_answer(response_text):
    # Clean the response text by removing excessive spaces
    response_clean = re.sub(r'\s+', ' ', response_text.strip())
    
    # Remove any "The answer is:" prefix if it exists
    response_clean = response_clean.replace("The answer is:", "").strip()

    # Extract answer directly after "####" in the response (this is common in many of your examples)
    match = re.search(r'####\s*(\S+)', response_clean)
    if match:
        return match.group(1), response_clean
    
    # Try extracting boxed answers (for cases where it's used)
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_clean)
    if boxed_match:
        boxed_content = boxed_match.group(1).strip()
        return boxed_content, response_clean
    
    # Fallback: if no special pattern found, return the response as it is
    return response_clean, response_clean


def extract_ground_truth(query_text):
    match = re.search(r"####\s*(\S+)", query_text)
    if match:
        ground_truth = match.group(1).strip().lower()
        return ground_truth
    return None

# The compute_rewards function
def compute_rewards(prompts, completions, **kwargs):
    rewards = []
    for query, response in zip(prompts, completions):
        ground_truth = extract_ground_truth(query)
        extracted, _ = extract_final_answer(response)

        #print(f"Extracted: {extracted}")  # Debugging
        #print(f"Ground Truth: {ground_truth}")  # Debugging

        if extracted == "empty":
            reward = 0.0
        else:
            # Clean and compare
            extracted_clean = extracted.strip().lower()  # Make it case-insensitive and remove spaces
            ground_truth_clean = ground_truth.strip().lower()

            if extracted_clean == ground_truth_clean:
                reward = 1.0
            else:
                reward = 0.0

        rewards.append(reward)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

    # Ensure the tensor is 1D
    if rewards_tensor.ndimension() > 1:
        rewards_tensor = rewards_tensor.squeeze()

    return rewards_tensor


# Function to test the model's prediction with ground truth
def test_model(model, tokenizer, dataset, num_samples=5):
    model.eval()
    for i in range(num_samples):
        data = dataset[i]
        prompt = data["prompt"]
        ground_truth = extract_ground_truth(data["answer"])

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response from model
        with torch.no_grad():
            output = model.generate(input_ids=inputs['input_ids'], max_length=256, num_return_sequences=1)
        
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract the answer from the model's response
        extracted_answer, _ = extract_final_answer(response_text)

        # Print the results
        print(f"Prompt: {prompt}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Model Output: {response_text}")
        print(f"Extracted Answer: {extracted_answer}")
        print("="*50)

# Training Arguments for GRPO
training_args = GRPOConfig(
    output_dir=OUTPUT_RL,
    learning_rate=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
)

# Setup GRPO trainer
grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=[compute_rewards],  # Ensure you have the compute_rewards function defined as per your needs
    args=training_args,
    train_dataset=dataset,
)

# Start GRPO training
print("Starting GRPO training...")
grpo_trainer.train()
print("GRPO training complete.")

grpo_trainer.save_model(OUTPUT_RL)
tokenizer.save_pretrained(OUTPUT_RL)

# After training, let's test the model's predictions
print("Testing the model's predictions with sample queries:")
test_model(model, tokenizer, dataset)

