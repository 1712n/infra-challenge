import asyncio

from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI
from transformers import pipeline
from starlette.requests import Request


app = FastAPI()
SHARED_PROCESS_POOL = ProcessPoolExecutor(max_workers=5)

MODELS = {
    "cardiffnlp": "models/twitter-xlm-roberta-base-sentiment",
    "ivanlau": "models/language-detection-fine-tuned-on-xlm-roberta-base",
    "svalabs": "models/twitter-xlm-roberta-crypto-spam",
    "EIStakovskii": "models/xlm_roberta_base_multilingual_toxicity_classifier_plus",
    "jy46604790": "models/Fake-News-Bert-Detect"
}


async def model_inference_task(model_path, q):
    print(f"loading {model_path}...")
    text_classification_pipeline = pipeline('text-classification', model=model_path, device=0)
    print(f"done loading {model_path}, waiting on the channel")
    while True:
        (text, response_q) = await q.get()
        inference_result = text_classification_pipeline(text)[0]
        await response_q.put(inference_result)


@app.on_event("startup")
async def startup_event():
    app.model_queues = {}
    for model_key, model_path in MODELS.items():
        q = asyncio.Queue()
        app.model_queues[model_key] = q
        asyncio.create_task(model_inference_task(model_path, q))


@app.post("/process")
async def process(request: Request):
    text = (await request.body()).decode()
    if not text:
        return {}
    result = {k: None for k in MODELS.keys()}
    for model_key, model_q in request.app.model_queues.items():
        response_q = asyncio.Queue()
        await model_q.put((text, response_q))
        result[model_key] = await response_q.get()
        if model_key == "cardiffnlp":
            result[model_key]["label"] = result[model_key]["label"].upper()
    return result 
