import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model path
BASE_MODEL_PATH = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"

# Load model and tokenizer from the same checkpoint
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Ensure that the tokenizer and model are compatible
if model.config.pad_token_id == model.config.eos_token_id:
    # Handle case where pad_token_id and eos_token_id are the same
    print("Warning: Pad token ID is the same as EOS token ID. This may cause unexpected behavior.")
    model.config.pad_token_id = None  # Or set to a specific token if you want

# Example text input
text = "What is the difference between simple and compound interest?"

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50)

# Manually create attention_mask if needed (default to all ones if not provided)
if 'attention_mask' not in inputs:
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

# Ensure pad_token_id is set
if model.config.pad_token_id is None:
    # If no pad token ID is set, you can use the EOS token or any specific token
    model.config.pad_token_id = tokenizer.eos_token_id

# Print tokenized input and attention mask
print(f"Tokenized Input: {inputs['input_ids']}")
print(f"Attention Mask: {inputs['attention_mask']}")

# Set generation parameters
max_new_tokens = 100  # Define how many new tokens you want to generate
max_length = 50  # Total length of input + generated tokens

# Ensure only one of max_length or max_new_tokens is set
# Here, we'll use max_new_tokens to control the number of generated tokens
output = model.generate(inputs['input_ids'], max_length=max_length, max_new_tokens=max_new_tokens)

# Decode the generated tokens
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print("Generated Text:", generated_text)

