import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nllb_translator")

app = FastAPI(
    title="NLLB Translation Service",
    description="Translation service using Meta's NLLB-200-1.3B model",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translation: str

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
    
    # Set the source language for the tokenizer
    tokenizer.src_lang = src_lang
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=256
        )
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info("Translation complete: %s", translated)
    return {"translation": translated}

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up FastAPI server...")
    load_model()
    logger.info("Server ready!")

@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    """FastAPI endpoint for translation"""
    try:
        if not all([req.text, req.src_lang, req.tgt_lang]):
            raise HTTPException(status_code=400, detail="Missing required fields: text, src_lang, or tgt_lang")
        
        result = perform_translation(req.text, req.src_lang, req.tgt_lang)
        return TranslationResponse(translation=result["translation"])
        
    except Exception as e:
        logger.error("Error in translation: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "NLLB Translation Service",
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "NLLB Translation Service", 
        "model": "facebook/nllb-200-1.3B",
        "endpoints": {
            "translate": "/translate",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "rp_handler:app",
        host="0.0.0.0",
        port=8000,
        workers=1
    ) 