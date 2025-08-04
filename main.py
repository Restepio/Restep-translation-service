import logging
import json
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nllb_translator")

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

# Load model
model_name = "facebook/nllb-200-3.3B"

try:
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded successfully")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Load model with safetensors to avoid security vulnerabilities
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        use_safetensors=True
    )
    logger.info("Model loaded successfully")
    
    # Move model to device after loading
    model = model.to(device)
    logger.info("Model moved to device: %s", device)
    
    model.eval()
    logger.info("Model set to eval mode")
    
except Exception as e:
    logger.error("Error during model loading: %s", str(e))
    raise e

# Shared translation function for both FastAPI and RunPod handler
def perform_translation(text: str, src_lang: str, tgt_lang: str):
    try:
        logger.info("Received translation request: %s -> %s", src_lang, tgt_lang)
        inputs = tokenizer(text, return_tensors="pt", src_lang=src_lang)
        
        # Move inputs to the same device as the model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=256
            )
        translated = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Translation complete: %s", translated)
        return {"translation": translated}
    except Exception as e:
        logger.error("Error during translation: %s", str(e))
        return {"error": str(e)}

# FastAPI endpoint
@app.post("/translate")
def translate(req: TranslationRequest):
    return perform_translation(req.text, req.src_lang, req.tgt_lang)

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "model": model_name}

# RunPod serverless handler
def handler(event):
    """
    RunPod serverless handler function.
    Expects event["input"] to contain the translation request.
    """
    try:
        logger.info("RunPod handler received event")
        
        # Extract input from event
        input_data = event.get("input", {})
        
        # Validate required fields
        if not all(key in input_data for key in ["text", "src_lang", "tgt_lang"]):
            return {
                "error": "Missing required fields. Need: text, src_lang, tgt_lang"
            }
        
        # Perform translation
        result = perform_translation(
            input_data["text"],
            input_data["src_lang"], 
            input_data["tgt_lang"]
        )
        
        logger.info("RunPod handler completed successfully")
        return result
        
    except Exception as e:
        logger.error("Error in RunPod handler: %s", str(e))
        return {"error": str(e)} 