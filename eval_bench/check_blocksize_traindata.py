#!/usr/bin/env python3
import json
from transformers import AutoTokenizer

# Update these paths as needed.
TRAIN_JSON_PATH = "/home/n/Token-Book/eval/Benchmark/MathQA/train.json"
OUTPUT_LOW_JSON_PATH = "/home/n/Token-Book/eval/Benchmark/MathQA/train_low_250.json"
OUTPUT_HIGH_JSON_PATH = "/home/n/Token-Book/eval/Benchmark/MathQA/train_high_250.json"
BASE_MODEL_PATH = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"

# Token count threshold for splitting.
TOKEN_THRESHOLD = 250

def main():
    # Load the tokenizer from your base model.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the training data.
    with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    low_token_examples = []
    high_token_examples = []
    token_counts = []
    total_examples = len(data)

    # Iterate over each example in the dataset.
    for idx, entry in enumerate(data):
        # Construct the prompt similar to your training setup.
        prompt = (
            f"Problem: {entry['Problem'].strip()}\n"
            f"Options: {entry['options'].strip()}\n"
            "Please explain your reasoning step-by-step and then provide your final answer.\n"
        )
        # Clean and prepare the rationale.
        rationale = entry.get("Rationale", "").strip()
        if rationale.startswith('"') and rationale.endswith('"'):
            rationale = rationale[1:-1].strip()
        target = f"Chain-of-Thought: {rationale}\nAnswer: {entry['correct'].strip()}\n"
        full_text = prompt + target

        # Tokenize the text.
        tokens = tokenizer.encode(full_text)
        count = len(tokens)
        token_counts.append(count)
        print(f"Example {idx+1}: {count} tokens")

        # Split based on the token threshold.
        if count <= TOKEN_THRESHOLD:
            low_token_examples.append(entry)
        else:
            high_token_examples.append(entry)
    
    # Optionally, print summary statistics.
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        print("\nSummary:")
        print(f"  Total examples in original file: {total_examples}")
        print(f"  Total examples with <= {TOKEN_THRESHOLD} tokens: {len(low_token_examples)}")
        print(f"  Total examples with > {TOKEN_THRESHOLD} tokens: {len(high_token_examples)}")
        print(f"  Average tokens per example: {avg_tokens:.1f}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Min tokens: {min(token_counts)}")

    # Write the low-token examples to a new JSON file.
    with open(OUTPUT_LOW_JSON_PATH, 'w', encoding='utf-8') as out_file:
        json.dump(low_token_examples, out_file, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(low_token_examples)} examples to {OUTPUT_LOW_JSON_PATH}")

    # Write the high-token examples to another JSON file.
    with open(OUTPUT_HIGH_JSON_PATH, 'w', encoding='utf-8') as out_file:
        json.dump(high_token_examples, out_file, indent=2, ensure_ascii=False)
    print(f"Saved {len(high_token_examples)} examples to {OUTPUT_HIGH_JSON_PATH}")

if __name__ == "__main__":
    main()

