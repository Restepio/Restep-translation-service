import logging
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

# Load model from the cached directory
model_name = "facebook/nllb-200-distilled-600M"
cache_dir = "/app/model_cache"

try:
    logger.info("Loading tokenizer and model from: %s", cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    logger.info("Tokenizer loaded successfully")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Load model with proper device mapping and low_cpu_mem_usage to avoid meta tensor issues
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        use_safetensors=True,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    logger.info("Model loaded successfully with device mapping")
    
    model.eval()
    logger.info("Model set to eval mode")
    
except Exception as e:
    logger.error("Error during model loading: %s", str(e))
    raise e

@app.post("/translate")
def translate(req: TranslationRequest):
    try:
        logger.info("Received translation request: %s -> %s", req.src_lang, req.tgt_lang)
        inputs = tokenizer(req.text, return_tensors="pt", src_lang=req.src_lang)
        
        # Move inputs to the same device as the model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[req.tgt_lang],
                max_length=256
            )
        translated = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Translation complete: %s", translated)
        return {"translation": translated}
    except Exception as e:
        logger.error("Error during translation: %s", str(e))
        return {"error": str(e)} 