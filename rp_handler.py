import runpod
import logging
from main import perform_translation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nllb_translator")

def handler(job):
    """
    Handler function that will be called by the serverless worker.
    Uses the shared perform_translation function from main.py
    """
    try:
        logger.info("RunPod handler received job")
        
        job_input = job['input']
        text = job_input.get('text')
        src_lang = job_input.get('src_lang')
        tgt_lang = job_input.get('tgt_lang')

        if not all([text, src_lang, tgt_lang]):
            return {"error": "Missing required fields: text, src_lang, or tgt_lang"}

        # Use the shared translation function from main.py
        result = perform_translation(text, src_lang, tgt_lang)
        
        logger.info("RunPod handler completed successfully")
        return result
        
    except Exception as e:
        logger.error("Error in RunPod handler: %s", str(e))
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 