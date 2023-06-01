import tornado.ioloop
import tornado.gen
import json
import redis
import logging
from os import environ

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REDIS_PASSWORD = environ.get("REDIS_PASSWORD")
REDIS_HOST = environ.get("REDIS_HOST")

# Подключение к Redis
r = redis.Redis(host=REDIS_HOST, password=REDIS_PASSWORD, port=6379, db=6)

async def result_listener(futures_dict):
    while True:
        pipeline = r.pipeline()
        logger.info('Connected to Redis Pipeline.')

        valid_sets = []  # Здесь мы будем хранить все наборы данных, которые удовлетворяют нашим условиям

        for key in r.scan_iter():
            # Здесь добавляем команду в пайплайн, но еще не выполняем ее
            pipeline.lrange(key, 0, -1)

        # Теперь выполняем все команды, добавленные в пайплайн
        values_list = pipeline.execute()

        for key, values in zip(r.scan_iter(), values_list):
            # Проверяем, что количество элементов в списке больше или равно 5
            if len(values) < 5:
                continue
            
            workers = set(json.loads(value.decode())['worker'] for value in values)

            # Проверяем, что результаты от всех воркеров присутствуют
            if len(workers) < 5:
                continue

            valid_sets.append((key, values))

        if valid_sets:
            for key, values in valid_sets:
                key = key.decode()
                final_message = {"correlation_id": key, "results": []}

                for value in values:
                    message = json.loads(value.decode())
                    final_message["results"].append({"worker": message["worker"], "result": message["result"]})

                # Добавляем команду в пайплайн и сразу выполняем
                pipeline.ltrim(key, len(values), -1).execute()

                future = futures_dict.pop(final_message["correlation_id"], None)
                if future is not None:
                    future.set_result(final_message)
                    logger.info(f'Successfully returned result for key: {key}')
                else:
                    logger.warning(f"No Future found for key: {key}. Current futures_dict: {futures_dict}")
        else:
            logger.info('No sets with 5 values found, skipping connection attempt.')
        
        await tornado.gen.sleep(0.01)
