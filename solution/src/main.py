import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from transformers import pipeline

from api.v1 import health_check, process
from core.config import settings
from ml_models import init_models as ml_models
from models.model_process import ModelPipeline

app = FastAPI(
    title=settings.project_name,
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    default_response_class=ORJSONResponse,
)


@app.on_event("startup")
async def startup():
    # load models
    if ml_models.models is None:
        ml_models.models = []
    for model in settings.MODELS:
        clf = pipeline(
            "text-classification",
            model=model.model_path,
            tokenizer=model.model_path,
            device=settings.device,
        )
        ml_models.models.append(ModelPipeline(pipeline=clf, model_process=model))
    app.include_router(
        health_check.router, prefix="/api/v1/health_check", tags=["Health Check"]
    )


@app.on_event("shutdown")
async def shutdown():
    ...


app.include_router(
    process.router,
    prefix="/api/v1/process",
    tags=["Process Model API"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
    )
