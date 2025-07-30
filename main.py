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
model_name = "facebook/nllb-200-1.3B"
cache_dir = "/app/model_cache"

logger.info("Loading tokenizer and model from: %s", cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)
model.to(device)
model.eval()
logger.info("Model loaded and ready")

@app.post("/translate")
def translate(req: TranslationRequest):
    logger.info("Received translation request: %s -> %s", req.src_lang, req.tgt_lang)
    inputs = tokenizer(req.text, return_tensors="pt", src_lang=req.src_lang).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[req.tgt_lang],
            max_length=256
        )
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info("Translation complete: %s", translated)
    return {"translation": translated}

