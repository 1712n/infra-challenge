from fastapi import FastAPI
from starlette.requests import Request

import models


pipelines = models.get_pipelines()
app = FastAPI()


@app.post("/process")
async def predict(request: Request):
    text = (await request.body()).decode()
    return await pipelines.process(text)