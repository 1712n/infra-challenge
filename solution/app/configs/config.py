from typing import List

from pydantic_yaml import YamlModel


class ModelConfig(YamlModel):
    model: str
    triton_model_name: str
    model_labels: List[str]


class AppConfig(YamlModel):
    # model parameters
    models: List[ModelConfig]
    # app parameters
    timeout: float
    triton_url: str
    batch_size: int

