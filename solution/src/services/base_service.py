from abc import ABC

from models.model_process import ModelPipeline


class BaseService(ABC):
    def __init__(self, models: list[ModelPipeline]):
        self.models = models
