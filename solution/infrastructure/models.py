from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import List

import torch
from transformers import pipeline


@dataclass
class TextClassificationModelData:
    model_name: str
    label: str
    score: float


class BaseTextClassificationModel(ABC):

    def __init__(self, name: str, model_path: str, tokenizer: str):
        self.name = name
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.device = 0 if torch.cuda.is_available() else -1
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Callable:
        ...

    @abstractmethod
    def __call__(self, input_texts: List[str]) -> List[TextClassificationModelData]:
        ...


class TransformerTextClassificationModel(BaseTextClassificationModel):

    def _load_model(self):
        sentiment_task = pipeline(
                "sentiment-analysis",
                model=self.model_path,
                tokenizer=self.model_path,
                device=self.device
                )
        return sentiment_task

    def __call__(self, input_texts: List[str]) -> List[TextClassificationModelData]:
        predictions = self.model(input_texts, batch_size=len(input_texts))
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return predictions

