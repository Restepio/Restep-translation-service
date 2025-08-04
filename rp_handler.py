import runpod
import logging
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nllb_translator")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer, device
    
    if model is None:
        model_name = "facebook/nllb-200-1.3B"
        cache_dir = "/app/model_cache"
        
        logger.info("Loading tokenizer and model...")
        
        # Try to load from cache first, fallback to downloading
        # Use safetensors to avoid PyTorch security vulnerability
        try:
            if os.path.exists(cache_dir):
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    use_safetensors=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    use_safetensors=True
                )
        except:
            # Fallback to downloading if cache fails
            logger.info("Cache not found, downloading model from Hugging Face Hub")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                use_safetensors=True
            )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        model.to(device)
        model.eval()
        logger.info("Model loaded and ready")

def perform_translation(text: str, src_lang: str, tgt_lang: str):
    """Translation function"""
    load_model()  # Ensure model is loaded
    
    logger.info("Received translation request: %s -> %s", src_lang, tgt_lang)
    inputs = tokenizer(text, return_tensors="pt", src_lang=src_lang).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=256
        )
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info("Translation complete: %s", translated)
    return {"translation": translated}

def handler(job):
    """
    Handler function that will be called by the serverless worker.
    """
    try:
        logger.info("RunPod handler received job")
        
        job_input = job['input']
        text = job_input.get('text')
        src_lang = job_input.get('src_lang')
        tgt_lang = job_input.get('tgt_lang')

        if not all([text, src_lang, tgt_lang]):
            return {"error": "Missing required fields: text, src_lang, or tgt_lang"}

        result = perform_translation(text, src_lang, tgt_lang)
        
        logger.info("RunPod handler completed successfully")
        return result
        
    except Exception as e:
        logger.error("Error in RunPod handler: %s", str(e))
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 