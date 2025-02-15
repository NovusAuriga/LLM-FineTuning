from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import json

##### DOWNLOAD MODEL ######
model_name = "Qwen/Qwen2.5-Math-1.5B"

# Force a redownload of the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)
print("Is Encoder-Decoder?", config.is_encoder_decoder)

## Specify your local model directory.
#local_model_path = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B"
## Load the configuration as a dictionary.
#config_path = os.path.join(local_model_path, "config.json")
#with open(config_path, "r", encoding="utf-8") as f:
#    config_dict = json.load(f)
#
## Inject the model_type key if it doesn't exist.
#if "model_type" not in config_dict:
#    config_dict["model_type"] = "qwen2"  # or choose a recognized type like "gpt2" if appropriate
#
## Create a configuration object from the dictionary.
#config = AutoConfig.from_dict(config_dict)
#
## Now load the model and tokenizer using the local path and the modified config.
#model = AutoModelForCausalLM.from_pretrained(local_model_path, config=config, trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
#
#print("Is Encoder-Decoder?", config.is_encoder_decoder)
#
