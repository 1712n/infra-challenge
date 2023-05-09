import time

from fastapi import FastAPI
from transformers import pipeline
from starlette.requests import Request

app = FastAPI()

models = {
    "cardiffnlp": {"path": "models/twitter-xlm-roberta-base-sentiment"},
    "ivanlau": {"path": "models/language-detection-fine-tuned-on-xlm-roberta-base"},
    "svalabs": {"path": "models/twitter-xlm-roberta-crypto-spam"},
    "EIStakovskii": {"path": "models/xlm_roberta_base_multilingual_toxicity_classifier_plus"},
    "jy46604790": {"path": "models/Fake-News-Bert-Detect"}
}

for model in models.values():
    model["pipeline"] = pipeline('text-classification', model=model["path"], device=0)


# todo: put each model into separate process and execute inference asynchronously
@app.post("/process")
async def process(request: Request):
    text = (await request.body()).decode()
    if not text:
        return {}
    result = {k: None for k in models.keys()}
    # start_overall = time.time()
    for model_key, model_value in models.items():
        print(f"Inference using model: {model}")
        start = time.time()
        result[model_key] = model_value["pipeline"](text)[0]
        elapsed = time.time() - start
        print(f"The result of the inference using {model_key} is {result}. It took {elapsed} seconds to compute.\n")
    # elapsed_overall = time.time() - start_overall
    # print(f"ELAPSED OVERALL: {elapsed_overall}")
    return result 


# async def server_loop(q):
#     pipe = pipeline(model="")
#     while True:
#         (string, response_q) = await q.get()
#         out = pipe(string)
#         await response_q.put(out)


# @app.on_event("startup")
# async def startup_event():
#     q = asyncio.Queue()
#     app.model_queue = q
#     asyncio.create_task(server_loop(q))