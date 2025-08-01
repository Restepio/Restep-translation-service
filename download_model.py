import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "facebook/nllb-200-1.3B"
print(f"Downloading model {model_name}...")

# Set a shared cache directory
cache_dir = "/app/model_cache"
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
)

# Save the model with safetensors
model.save_pretrained(cache_dir, safe_serialization=True)
tokenizer.save_pretrained(cache_dir)

print("Model downloaded and saved with safetensors successfully.") 