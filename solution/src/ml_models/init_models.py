from models.model_process import ModelPipeline

models: list[ModelPipeline] = None


async def load_models() -> list[ModelPipeline]:
    return models
