from typing import List
import asyncio

from pydantic import ValidationError

from infrastructure.models import TextClassificationModelData
from service.recognition import TextClassificationService
from handlers.data_models import ResponseSchema, RecognitionSchema


class PredictionHandler:

    def __init__(self, recognition_service: TextClassificationService, timeout: float):
        self.recognition_service = recognition_service
        self.timeout = timeout

    async def tokenize_texts_batch(self, consumer_queue, producer_queues, max_batch_size: int):
        while True:
            texts = []
            queues = []

            try:
                while True:
                    text, response_q = await asyncio.wait_for(consumer_queue.get(), timeout=self.timeout)
                    texts.append(text)
                    queues.append(response_q)

            except asyncio.exceptions.TimeoutError:
                pass

            if texts:
                for text_batch in self._perform_batches(texts, max_batch_size):
                    inputs = self.recognition_service.service_models[0].tokenize_texts(text_batch)

                    for output_queue in producer_queues:
                        await output_queue.put((inputs, queues))

    async def handle(self, model_name, model_queue):
        while True:
            inputs = None
            queues = []

            while True:
                inputs, queues = await model_queue.get()

                if inputs:
                    model = next(
                            (model for model in self.recognition_service.service_models if model.name == model_name),
                            None
                            )
                    if model:
                        outs = model(inputs)
                        for rq, out in zip(queues, outs):
                            await rq.put(out)

    def serialize_answer(self, results: List[TextClassificationModelData]) -> ResponseSchema:
        res_model = {rec.model_name: self._recognitions_to_schema(rec) for rec in results}
        return ResponseSchema(**res_model)

    def _recognitions_to_schema(self, recognition: TextClassificationModelData) -> RecognitionSchema:
        if recognition.model_name != "ivanlau":
            recognition.label = recognition.label.upper()
        return RecognitionSchema(score=recognition.score, label=recognition.label)

    def _perform_batches(self, texts: List[str], max_batch_size):
        for i in range(0, len(texts), max_batch_size):
            yield texts[i:i + max_batch_size]

