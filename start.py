#!/usr/bin/env python3
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("startup")

def main():
    """
    Startup script that detects the environment and runs the appropriate service:
    - If PORT environment variable is set, run FastAPI server
    - Otherwise, run RunPod serverless handler
    """
    port = os.environ.get('PORT')
    
    if port:
        logger.info(f"PORT environment variable detected ({port}). Starting FastAPI server...")
        
        # Import and run FastAPI server
        import uvicorn
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=int(port),
            workers=1
        )
    else:
        logger.info("No PORT environment variable. Starting RunPod serverless handler...")
        
        # Import and run RunPod handler
        import runpod
        from rp_handler import handler
        runpod.serverless.start({"handler": handler})

if __name__ == "__main__":
    main()