from transformers import AutoConfig

BASE_MODEL_PATH = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"

# Load the model's configuration
config = AutoConfig.from_pretrained(BASE_MODEL_PATH)

# Print the configuration to inspect the architecture type
print(config)

