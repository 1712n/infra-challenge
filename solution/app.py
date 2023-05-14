import asyncio

from fastapi import FastAPI
from transformers import pipeline
from starlette.requests import Request

app = FastAPI()

models = {
    "cardiffnlp": {"name": "cardiffnlp/twitter-xlm-roberta-base-sentiment"},
    "ivanlau": {"name": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base"},
    "svalabs": {"name": "svalabs/twitter-xlm-roberta-crypto-spam"},
    "EIStakovskii": {"name": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus"},
    "jy46604790": {"name": "jy46604790/Fake-News-Bert-Detect"}
}

for model_key, model_dict in models.items():
    text_classification_pipeline = pipeline('text-classification', model=model_dict["name"], device="cuda:0")
    model_dict["pipeline"] = text_classification_pipeline


async def model_inference_task(model_name, q):
    while True:
        strings, queues = [], []
        while True:
            try:
                (string, rq) = await asyncio.wait_for(q.get(), timeout=0.015)
            except asyncio.exceptions.TimeoutError:
                break
            strings.append(string)
            queues.append(rq)
            if len(strings) == 3:
                break
        if not strings:
            continue
        outs = models[model_name]["pipeline"](strings, batch_size=len(strings))
        for rq, out in zip(queues, outs):
            await rq.put(out)


@app.on_event("startup")
async def startup_event():
    app.model_queues = {}
    for model_key in models.keys():
        q = asyncio.Queue()
        app.model_queues[model_key] = q
        asyncio.create_task(model_inference_task(model_key, q))


@app.post("/process")
async def process(request: Request):
    text = (await request.body()).decode()
    if not text:
        return {}
    result = {k: None for k in models.keys()}
    for model_key, model_q in request.app.model_queues.items():
        response_q = asyncio.Queue()
        await model_q.put((text, response_q))
        result[model_key] = await response_q.get()
        if model_key == "cardiffnlp":
            result[model_key]["label"] = result[model_key]["label"].upper()
    return result 
