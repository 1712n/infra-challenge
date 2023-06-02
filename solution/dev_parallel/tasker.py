import asyncio
import aioredis
import json
import logging
import os
from config import config  # импортируем конфигурационный файл

logging.basicConfig(level=logging.INFO)

class QueueManager:
    def __init__(self):
        self.redis_host = os.environ.get('REDIS_HOST')
        self.redis_port = 6379
        self.redis_password = os.environ.get('REDIS_PASSWORD')
        self.group_settings = self.load_group_settings()  # загружаем настройки групп
        self.lock = asyncio.Lock()

    def load_group_settings(self):
        # загружаем конфигурацию групп только один раз при инициализации
        group_settings = []
        for key in config.keys():
            if 'farm_' in key:
                group_settings.append({'db_numbers': config[key]['db']})
        return group_settings

    async def start(self):
        logging.info('Starting Queue Manager...')
        self.redis_source = aioredis.from_url(f"redis://{self.redis_host}:{self.redis_port}/0", password=self.redis_password)
        self.target_groups = [
            [aioredis.from_url(f"redis://{self.redis_host}:{self.redis_port}/{i}", password=self.redis_password) for i in group['db_numbers']]
            for group in self.group_settings
        ]
        logging.info('Connected to Redis.')
        await self.transfer_messages_from_redis_to_redis()

    async def publish_message_to_redis(self, message_body, redis_db):
        await redis_db.rpush('dispatcher', message_body)

    async def transfer_messages_from_redis_to_redis(self, redis_queue_name='dispatcher'):
        current_group_index = 0  # Индекс текущей группы
        while True:
            async with self.lock:
                logging.info('Waiting for message in Redis queue...')
                _, message_body = await self.redis_source.blpop(redis_queue_name)
                logging.info(f'Received message from Redis queue: {message_body}')
                if message_body:
                    target_group = self.target_groups[current_group_index]  # Выбираем текущую группу
                    logging.info(f'Sending message to group {current_group_index}.')
                    publish_tasks = [self.publish_message_to_redis(message_body, redis_target) for redis_target in target_group]
                    await asyncio.gather(*publish_tasks)
                    logging.info(f'Published message to group {current_group_index}.')
                    current_group_index = (current_group_index + 1) % len(self.target_groups)  # Переходим к следующей группе
                else:
                    logging.error('Received empty message from Redis queue.')
                await asyncio.sleep(0.01)  # Добавляем небольшую задержку, чтобы уменьшить нагрузку на ЦП

loop = asyncio.get_event_loop()
manager = QueueManager()
loop.run_until_complete(manager.start())
