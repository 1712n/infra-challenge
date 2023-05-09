from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any
from app.model.model import predict_pipeline

app = FastAPI()

class TextIn(BaseModel):
    str

class ModelOut(BaseModel):
    label: str
    score: float

class PredictionOut(BaseModel):
    cardiffnlp: ModelOut
    ivanlau: ModelOut
    svalabs: ModelOut
    EIStakovskii: ModelOut
    jy46604790: ModelOut

@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict", response_model=PredictionOut)
async def predict(request: str = Body(...)):
    result = await predict_pipeline(request)
    return result