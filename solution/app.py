import time

from fastapi import FastAPI
from transformers import pipeline
from starlette.requests import Request

app = FastAPI()

models = {
    "cardiffnlp": {"path": "models/twitter-xlm-roberta-crypto-spam"},
    "ivanlau": {"path": "models/language-detection-fine-tuned-on-xlm-roberta-base"},
    "svalabs": {"path": "models/twitter-xlm-roberta-crypto-spam"},
    "EIStakovskii": {"path": "models/xlm_roberta_base_multilingual_toxicity_classifier_plus"},
    "jy46604790": {"path": "models/Fake-News-Bert-Detect"}
}

for model in models.values():
    model["pipeline"] = pipeline('text-classification', model=model["path"])


@app.post("/process")
async def process(request: Request):
    text = str(await request.body())
    result = {k: None for k in models.keys()}
    start_overall = time.time()
    for model_key, model_value in models.items():
        print(f"Inference using model: {model}")
        start = time.time()
        result[model_key] = model_value["pipeline"](text)[0]
        elapsed = time.time() - start
        print(f"The result of the inference using {model_key} is {result}. It took {elapsed} seconds to compute.\n")
    elapsed_overall = time.time() - start_overall
    print(f"ELAPSED OVERALL: {elapsed_overall}")
    return result 