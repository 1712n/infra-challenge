from typing import Optional

from pydantic import BaseModel
from transformers import TextClassificationPipeline


class ModelProcess(BaseModel):
    model_name: str
    model_path: str
    output_functions: Optional[list[str]] = None


class ModelPipeline(BaseModel):
    model_process: ModelProcess
    pipeline: TextClassificationPipeline

    class Config:
        arbitrary_types_allowed = True
