from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

from infrastructure.models import TritonTextClassificationModel, TextClassificationModelData


class TextClassificationService:

    def __init__(self, models: List[TritonTextClassificationModel]):
        self.service_models = models

