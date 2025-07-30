# NLLB-200 Translation Service for RunPod Serverless

This project provides a RunPod Serverless endpoint for translating text using Meta’s `facebook/nllb-200-1.3B` model, powered by Hugging Face Transformers.

## Usage

[![Runpod](https://api.runpod.io/badge/Restepio/Restep-translation-service)](https://console.runpod.io/hub/Restepio/Restep-translation-service)

Once deployed on RunPod, you can send translation requests to the serverless endpoint. The handler expects a JSON object with the following structure:

```json
{
  "input": {
    "text": "Your text to translate here.",
    "src_lang": "eng_Latn",
    "tgt_lang": "fra_Latn"
  }
}
```

- `text`: The string of text you want to translate.
- `src_lang`: The source language code (e.g., `eng_Latn` for English).
- `tgt_lang`: The target language code (e.g., `fra_Latn` for French).

### Example Response

The endpoint will return a JSON object containing the translation:

```json
{
  "translation": "Votre texte à traduire ici."
}
```