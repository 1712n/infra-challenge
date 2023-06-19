import logging
import json
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Worker:
    def __init__(self, worker_config, message_queue, results_queue, event, logger):
        self.logger = logger
        # Взять имя воркера из конфигурации
        self.worker_name = worker_config["worker_name"]
        self.model_name = worker_config["model_name"]
        self.model_labels = worker_config["model_labels"]
        self.event = event

        self.message_queue = message_queue
        self.results_queue = results_queue

        # Загрузка модели
        logger.info(f"Loading model {self.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            logger.info("Using GPU for model computations.")
        else:
            logger.info("Using CPU for model computations.")
        logger.info(f"Model {self.model_name} loaded.")
        
        # Загрузка токенизатора
        logger.info(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Tokenizer for {self.model_name} loaded.")

        # Проверка модели на тестовых данных
        self.test_model()

        # Signal that the worker has fully loaded
        self.event.set()
        logger.info(f"Worker started.{self.worker_name} ")

    def test_model(self):
        test_data = "This is a test sentence."
        inputs = self.tokenizer(test_data, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        while True:
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)

                predictions = outputs.logits.argmax(dim=-1).item()
                logger.info(f"Test data processed successfully. Predicted label: {predictions}")
                break  # Если тест прошел успешно, выходим из цикла
            except Exception as e:
                logger.error(f"Test failed with error: {e}. Retrying...")
                continue

    BATCH_SIZE = 5  # Подходящий размер пакета

    async def start(self):
        batch = []
        while True:
            if not self.message_queue.empty() and len(batch) < self.BATCH_SIZE:
                message = await self.message_queue.get()
                batch.append(message)
            elif batch:
                await self.process_batch(batch)
                batch = []
            else:
                await asyncio.sleep(0.01) 



    async def process_batch(self, batch):
        bodies = [json.loads(message) if isinstance(message, str) else message for message in batch]
        texts = [body['data']['data'] for body in bodies]
        correlation_ids = [body['correlation_id'] for body in bodies]
        
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')  # 

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for i, prediction in enumerate(predictions):
            score = probabilities[i][prediction].item()
            label = self.model_labels[prediction.item()]
            result_key = self.model_name.split('/')[0]
            result = {result_key: {"score": score, "label": label}}

            results_dict = {
                             "correlation_id": correlation_ids[i],
                             "worker": self.worker_name,
                             "result": result
                            }
            results_dict = json.dumps(results_dict)
            await self.results_queue.put(results_dict)

       
