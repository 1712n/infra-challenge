from fastapi import FastAPI, Request
from models import initialize_models, models
from inference import run_inference
import asyncio

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    try:
        await initialize_models()
    except RuntimeError as e:
        # Handle the initialization error and return an error response if needed
        return {"error": str(e)}


@app.post("/process")
async def process_text(request: Request):
    try:
        data = await request.json()
        text = data.strip('"')

        tasks = []
        for model_name in models.keys():
            task = asyncio.create_task(run_inference(model_name, text))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        response = {}
        for model_name, result in zip(models.keys(), results):
            response[model_name] = {
                "score": result["score"],
                "label": result["label"]
            }

        return response

    except Exception as e:
        # Handle any exceptions and return an error response
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="localhost", port=8000)
