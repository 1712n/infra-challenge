from cachetools import TTLCache
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..models import get_all_predictions
from .validation import Model

app = FastAPI()

get_all_predictions("Hello, World!")

# Create a cache
cache = TTLCache(maxsize=1000, ttl=600)

@app.post("/process", response_model=Model)
async def process(request: Request):
    text = await request.body()
    text = text.decode("utf-8")

    # Check if the result is already in the cache
    result = cache.get(text)
    if result:
        return JSONResponse(result)

    result = get_all_predictions(text)

    # Store the result in the cache
    cache[text] = result

    return JSONResponse(result)
