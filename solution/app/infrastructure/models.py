from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

import numpy as np
import tritonclient.http


@dataclass
class TextClassificationModelData:
    model_name: str
    label: str
    score: float


class TritonTextClassificationModel:

    def __init__(self, name: str, triton_model_name: str, model_url: str, batch_size: int, model_labels: List[str]):
        self.name = name
        self.triton_model_name = triton_model_name
        self.model_url = model_url
        self.batch_size = batch_size
        self.model_labels = model_labels
        self.model_version = 1
        self.triton_client = triton_client = tritonclient.http.InferenceServerClient(url=self.model_url, verbose=False)
        self._load_model()
        self.model_metadata = triton_client.get_model_metadata(
                model_name=self.triton_model_name,
                model_version=self.model_version
        )
        self.query = tritonclient.http.InferInput(name="TEXT", shape=(self.batch_size,), datatype="BYTES")
        self.model_score = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)

    def _load_model(self):
        assert self.triton_client.is_model_ready(
                model_name=self.triton_model_name, model_version=self.model_version), f"model {self.name} not yet ready"

    def __call__(self, texts: List[str]):
        predictions = []
        for text_batch in self._perform_batches(texts):
            self.query.set_data_from_numpy(np.asarray(text_batch, dtype=object))
            response = self.triton_client.infer(
                    model_name=self.triton_model_name,
                    model_version=self.model_version,
                    inputs=[self.query],
                    outputs=[self.model_score]
            )
            logits = response.as_numpy("output")
            batch_res = self._results_from_logits(logits)
            predictions.extend(batch_res)
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return  predictions

    def _results_from_logits(self, logits: np.ndarray):

        label_ids = logits.argmax(axis=1)
        scores = self._softmax(logits)
        results = [
                {
                    "label": self.model_labels[label_id],
                    "score": score
                }
                for label_id, score in zip(label_ids, scores)
            ]
        return results

    def _perform_batches(self, texts: List[str]):
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def _softmax(self, logits) -> List[float]:
        scores = []
        for x in logits:
            e_x = np.exp(x - np.max(x))
            scores.append(np.amax(e_x / e_x.sum(axis=0)))
        return scores

