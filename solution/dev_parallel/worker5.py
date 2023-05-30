import logging
import json
import redis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Import config from config.py
from config import config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Worker:
    def __init__(self, worker_config, connection_config):
        # получение значений переменных окружения
        redis_host = os.environ.get('REDIS_HOST')
        redis_password = os.environ.get('REDIS_PASSWORD')
        redis_port = int(os.environ.get('REDIS_PORT', '6379')) 

        self.redis_result_config = connection_config["results_db"]

        # Взять имя воркера из имени файла
        self.queue_name = os.path.splitext(os.path.basename(__file__))[0]
        self.model_name = worker_config["model_name"]
        self.model_labels = worker_config["model_labels"]

        self.redis_incoming = redis.Redis(host=redis_host, 
                                          port=redis_port, 
                                          db=worker_config['db'], 
                                          password=redis_password)
        
        self.redis_outgoing = redis.Redis(host=redis_host, 
                                          port=redis_port, 
                                          db=self.redis_result_config['db'], 
                                          password=redis_password)

        # Загрузка модели
        logger.info(f"Loading model {self.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        logger.info(f"Model {self.model_name} loaded.")
        
        # Загрузка токенизатора
        logger.info(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Tokenizer for {self.model_name} loaded.")
    
    def start(self):
        logger.info('Worker started.')
        while True:
            _, message = self.redis_incoming.blpop('dispatcher')
            message_body = json.loads(message)
            self.process_message(message_body)

    def process_message(self, body):
        correlation_id = body['correlation_id']
        text = body['data']['data']
        logger.info(f"Processing text: {text}")
        inputs = self.tokenizer(text, return_tensors='pt')

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1).item()
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score = probabilities[0][predictions].item()
        
        label = self.model_labels[predictions]

        result_key = self.model_name.split('/')[-1]
        result = {result_key: {"score": score, "label": label}}
        logger.info(f"Received task results {result}")

        results_dict = {
                         "correlation_id": correlation_id,
                         "worker": self.queue_name,
                         "result": result[self.model_name.split('/')[-1]]
                        }
        results_dict = json.dumps(results_dict)
        self.redis_outgoing.rpush(correlation_id, results_dict)
        logger.info(f"Saved result to Redis with key {correlation_id} : {results_dict}")

# Загрузка конфигурации подключений из файла config.py
connection_config = config

worker_name = os.path.splitext(os.path.basename(__file__))[0]
worker_config = next((worker for worker in connection_config["workers"] if worker["worker_name"] == worker_name), None)

if not worker_config:
    raise ValueError(f"No configuration found for worker {worker_name}")

# Создание и запуск воркера
if __name__ == "__main__":
    worker = Worker(worker_config, connection_config)
    worker.start()
