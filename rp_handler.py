import runpod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model at startup
model_name = "facebook/nllb-200-1.3B"
cache_dir = "/app/model_cache"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()
print("Model loaded and ready")

def handler(job):
    """
    Handler function that will be called by the serverless worker.
    """
    job_input = job['input']
    text = job_input.get('text')
    src_lang = job_input.get('src_lang')
    tgt_lang = job_input.get('tgt_lang')

    if not all([text, src_lang, tgt_lang]):
        return {"error": "Missing required fields: text, src_lang, or tgt_lang"}

    print(f"Received translation request: {src_lang} -> {tgt_lang}")

    inputs = tokenizer(text, return_tensors="pt", src_lang=src_lang).to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=256
        )
    
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Translation complete: {translated}")
    
    return {"translation": translated}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 