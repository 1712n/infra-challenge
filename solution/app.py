import asyncio

import torch

from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from starlette.requests import Request
from optimum.onnxruntime import ORTModelForSequenceClassification

app = FastAPI()

MODELS = {
    "cardiffnlp": "/models/twitter-xlm-roberta-base-sentiment/",
    "ivanlau": "/models/language-detection-fine-tuned-on-xlm-roberta-base/",
    "svalabs": "/models/twitter-xlm-roberta-crypto-spam/",
    "EIStakovskii": "/models/xlm_roberta_base_multilingual_toxicity_classifier_plus/",
    "jy46604790": "/models/Fake-News-Bert-Detect/"
}


async def model_inference_task(model_name: str, q: asyncio.Queue):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ORTModelForSequenceClassification.from_pretrained(model_name, provider="CUDAExecutionProvider")
    model.to(device="cuda:0")

    while True:
        strings, queues = [], []
        while True:
            try:
                (string, rq) = await asyncio.wait_for(q.get(), timeout=0.025)
            except asyncio.exceptions.TimeoutError:
                break
            strings.append(string)
            queues.append(rq)
            if len(strings) == 8:
                break
        if not strings:
            continue

        encoded_input = tokenizer(
            strings,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
        ).to(device="cuda:0")
        logits = model(**encoded_input).logits

        id2label = model.config.id2label
        label_ids = logits.argmax(dim=1)
        scores = logits.softmax(dim=-1)
        outs = [
            {
                "label": id2label[label_id.item()],
                "score": score[label_id.item()].item()
            }
            for label_id, score in zip(label_ids, scores)
        ]

        for rq, out in zip(queues, outs):
            await rq.put(out)


@app.on_event("startup")
async def startup_event():
    app.model_queues = {}
    for model_key, model_name in MODELS.items():
        q = asyncio.Queue()
        app.model_queues[model_key] = q
        asyncio.create_task(model_inference_task(model_name, q))


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
