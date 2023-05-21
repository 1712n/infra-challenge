from typing import List

from pydantic_yaml import YamlModel


class ModelConfig(YamlModel):
    model: str
    model_path: str


class AppConfig(YamlModel):
    # model parameters
    tokenizer: str
    models: List[ModelConfig]
    # app parameters
    port: int
    workers: int
    # async queues parameters
    timeout: float

