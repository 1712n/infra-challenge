import asyncio
from models import initialized_models, models
from typing import Dict
from transformers import pipeline

# Run inference on a specific model
async def run_inference(model_name: str, text: str) -> Dict[str, float]:
    model = initialized_models[model_name]
    output = await asyncio.to_thread(model, text)
    score = output[0]['score']
    label = output[0]['label']
    return {"score": score, "label": label}
