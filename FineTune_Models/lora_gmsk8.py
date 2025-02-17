#!/usr/bin/python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig

from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset

# Set the paths
BASE_MODEL_PATH = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
OUTPUT_DIR = "checkpoints/qwen2.5-lora-0.5b"
LOCAL_DATA_PATH = "/home/n/Token-Book/deepseek_fine/datasets/gsm8k/processed/train_data.jsonl"

# Load the local dataset
dataset = Dataset.from_json(LOCAL_DATA_PATH)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Model setup with LoRA
model_config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
base_model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=2,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, lora_config)

# Map the dataset to include the necessary columns for the Trainer
def tokenize_function(examples):
    # Tokenize the prompt and answer
    encoding = tokenizer(examples['prompt'], truncation=True, padding="max_length", max_length=256)
    encoding['labels'] = encoding['input_ids'].copy()  # Labels should be the same as input_ids for language modeling
    return encoding

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="adamw_torch",
    evaluation_strategy="no",
    save_steps=200,
    logging_steps=10,
    learning_rate=1e-5,
    warmup_steps=20,
    fp16=True,
    seed=90134,
    max_grad_norm=1.0,
    lr_scheduler_type='linear',
    remove_unused_columns=False  # This is the key to prevent the error
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# Start training with LoRA
print("Starting LoRA training...")
trainer.train()
print("LoRA training complete.")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

