from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..models import get_all_predictions
from .validation import Model

app = FastAPI()

get_all_predictions("Hello, World!")


@app.post("/process", response_model=Model)
async def method(request: Request):
    text = await request.body()
    text = text.decode("utf-8")
    result = get_all_predictions(text)
    return JSONResponse(result)
