#!/usr/bin/env python3
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

# -----------------------------------------------------------------------------
# 1) CONFIGURE PATHS
# -----------------------------------------------------------------------------
# Path to the base model (from Hugging Face cache).
BASE_MODEL_PATH = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"

# Path to the adapter checkpoint directory.
ADAPTER_CHECKPOINT = "/home/n/Token-Book/deepseek_fine/checkpoints/qwen2.5-GRPO-GOOD"

# Directory to save the merged model.
MERGED_MODEL_DIR = "/home/n/Token-Book/deepseek_fine/checkpoints/Qwen2.5-GOOD"

# -----------------------------------------------------------------------------
# 2) SET UP MODEL AND ADAPTER CONFIGURATION
# -----------------------------------------------------------------------------
# Load the tokenizer (optional but useful for saving later)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model configuration and base model.
model_config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# Optionally, enable gradient checkpointing for memory efficiency.
base_model.gradient_checkpointing_enable()

# Set up the LoRA configuration.
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
)

# Wrap the base model with LoRA.
model = get_peft_model(base_model, lora_config)

# -----------------------------------------------------------------------------
# 3) LOAD ADAPTER WEIGHTS
# -----------------------------------------------------------------------------
# Here we assume your adapter weights were saved as a single file (e.g. adapter_model.safetensors)
adapter_weights_path = os.path.join(ADAPTER_CHECKPOINT, "adapter_model.safetensors")
if not os.path.exists(adapter_weights_path):
    raise FileNotFoundError(f"Adapter weights not found at: {adapter_weights_path}")
print("Loading adapter weights from:", adapter_weights_path)
adapter_weights = load_file(adapter_weights_path, device="cpu")

# Load the adapter weights into the PEFT-wrapped model.
model.load_state_dict(adapter_weights, strict=False)

# -----------------------------------------------------------------------------
# 4) MERGE THE ADAPTER WEIGHTS INTO THE BASE MODEL
# -----------------------------------------------------------------------------
# This merges the adapter weights into the base model weights and unloads the LoRA layers.
print("Merging adapter weights into the base model...")
merged_model = model.merge_and_unload()

# -----------------------------------------------------------------------------
# 5) SAVE THE MERGED MODEL
# -----------------------------------------------------------------------------
print("Saving the merged model to:", MERGED_MODEL_DIR)
merged_model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)
print("Merge complete.")

