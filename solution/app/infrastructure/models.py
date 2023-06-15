from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from time import sleep

import numpy as np
import tritonclient.http
from socket import error as socket_error


@dataclass
class TextClassificationModelData:
    model_name: str
    label: str
    score: float


class TritonTextClassificationModel:

    def __init__(self, name: str, triton_model_name: str, triton_url: str, batch_size: int, model_labels: List[str]):
        self.name = name
        self.triton_model_name = triton_model_name
        self.triton_url = triton_url
        self.batch_size = batch_size
        self.model_labels = model_labels
        self.model_version = "1"
        self.triton_client = triton_client = tritonclient.http.InferenceServerClient(url=self.triton_url, verbose=False)
        self._load_model()
        self.model_metadata = triton_client.get_model_metadata(
                model_name=self.triton_model_name,
                model_version=self.model_version
        )
        self.model_score = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)

    def _load_model(self):
        try:
            self.triton_client.is_model_ready(model_name=self.triton_model_name, model_version=self.model_version)
        except socket_error as err:
            sleep(10)
            self._load_model()

    def __call__(self, texts: List[str]):
        predictions = []
        query = tritonclient.http.InferInput(name="TEXT", shape=(self.batch_size,), datatype="BYTES")

        for text_batch in self._perform_batches(texts):
            if len(text_batch) != self.batch_size:
                query = tritonclient.http.InferInput(name="TEXT", shape=(len(text_batch),), datatype="BYTES")

            query.set_data_from_numpy(np.asarray(text_batch, dtype=object))
            batch_res = self._request_model(query)
            predictions.extend(batch_res)
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return  predictions

    def _request_model(self, query):
        response = self.triton_client.infer(
                model_name=self.triton_model_name,
                model_version=self.model_version,
                inputs=[query],
                outputs=[self.model_score]
        )
        logits = response.as_numpy("output")
        batch_res = self._results_from_logits(logits)
        return batch_res

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

