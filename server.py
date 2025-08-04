import logging
from fastapi import FastAPI
from pydantic import BaseModel
from rp_handler import perform_translation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("translation_server")

app = FastAPI(title="NLLB Translation Service", version="1.0.0")

class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

@app.post("/translate")
def translate(req: TranslationRequest):
    """FastAPI endpoint for translation"""
    return perform_translation(req.text, req.src_lang, req.tgt_lang)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "NLLB Translation Service"}

@app.get("/")
def root():
    """Root endpoint with basic info"""
    return {
        "message": "NLLB Translation Service", 
        "endpoints": {
            "translate": "/translate",
            "health": "/health"
        }
    }