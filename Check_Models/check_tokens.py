#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your pre-trained model
BASE_MODEL_PATH = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
TRAIN_JSON_PATH = "/home/n/Token-Book/eval/Benchmark/MathQA/40k_train_250.json"


# Load the tokenizer and model from the same checkpoint
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# -----------------------------------------------------------------------------
# Step 1: Ensure Tokenizer and Model Vocabulary Sizes Match
print("Checking if tokenizer and model vocabulary sizes match...")

tokenizer_vocab_size = len(tokenizer)
model_vocab_size = model.get_input_embeddings().weight.shape[0]

if tokenizer_vocab_size == model_vocab_size:
    print("Tokenizer and model vocabulary sizes match!")
else:
    print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
    print(f"Model vocabulary size: {model_vocab_size}")
    print("The tokenizer and model vocabulary sizes do not match!")

# -----------------------------------------------------------------------------
# Step 2: Verify Special Tokens Match
print("Checking if special token IDs match...")

# Get special token IDs from the tokenizer
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
unk_token_id = tokenizer.unk_token_id

# Compare with model config
assert pad_token_id == model.config.pad_token_id, "Pad token IDs do not match!"
assert eos_token_id == model.config.eos_token_id, "EOS token IDs do not match!"
assert unk_token_id == model.config.unk_token_id, "UNK token IDs do not match!"

print("Special token IDs match between tokenizer and model!")

# -----------------------------------------------------------------------------
# Step 3: Tokenize and Decode a Sample from Your Training JSON
print("Checking tokenization of a sample from the training JSON...")

# Load the training data
with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Choose a sample from the training JSON (e.g., the first entry)
sample = data[0]
text = f"Problem: {sample['Problem']}\nOptions: {sample['options']}\nPlease explain your reasoning step-by-step and then provide your final answer."

# Tokenize the sample text
tokens = tokenizer.encode(text)
decoded_text = tokenizer.decode(tokens)

print(f"Original text:\n{text}")
print(f"Tokenized IDs:\n{tokens}")
print(f"Decoded back:\n{decoded_text}")

# Ensure that decoding works correctly and matches the original text
assert text == decoded_text, "Decoded text does not match the original text!"

# -----------------------------------------------------------------------------
# Step 4: Verify Token IDs Alignment with Model's Input Embeddings
print("Verifying token IDs alignment with model's input embeddings...")

# Check the embeddings for the token IDs
embedding = model.get_input_embeddings()(torch.tensor(tokens))

# Verify that the token IDs map to valid embeddings
assert embedding.shape[0] == len(tokens), "Mismatch in token ID to embedding mapping!"
print(f"Successfully aligned {len(tokens)} token IDs with model embeddings!")

# -----------------------------------------------------------------------------
# Step 5: Compare Specific Tokens for Alignment
print("Checking specific token ID alignment...")

# Check token ID for a specific word or token
example_word = "Problem"
token_id_example = tokenizer.encode(example_word)[0]
print(f"Token ID for '{example_word}': {token_id_example}")

# Retrieve the model's embedding for the token ID
embedding_example = model.get_input_embeddings()(torch.tensor([token_id_example]))
print(f"Embedding for token ID {token_id_example}: {embedding_example.squeeze().shape}")

# -----------------------------------------------------------------------------
# Finished checks
print("All checks completed successfully.")

