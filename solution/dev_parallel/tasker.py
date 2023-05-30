import asyncio
import aioredis
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

class QueueManager:
    def __init__(self):
        self.redis_host = os.environ.get('REDIS_HOST')  
        self.redis_port = 6379 
        self.redis_password = os.environ.get('REDIS_PASSWORD')  

    async def start(self):
        logging.info('Starting Queue Manager...')
        self.redis_source = aioredis.from_url(f"redis://{self.redis_host}:{self.redis_port}/0", password=self.redis_password)
        self.redis_targets = [aioredis.from_url(f"redis://{self.redis_host}:{self.redis_port}/{i}", password=self.redis_password) for i in range(1, 6)]
        logging.info('Connected to Redis.')
        await self.transfer_messages_from_redis_to_redis()

    async def publish_message_to_redis(self, message_body, redis_db):
        await redis_db.rpush('dispatcher', message_body)

    async def transfer_messages_from_redis_to_redis(self, redis_queue_name='dispatcher'):
        while True:
            logging.info('Waiting for message in Redis queue...')
            _, message_body = await self.redis_source.blpop(redis_queue_name)
            logging.info(f'Received message from Redis queue: {message_body}')
            publish_tasks = [self.publish_message_to_redis(message_body, redis_target) for redis_target in self.redis_targets]
            await asyncio.gather(*publish_tasks)
            logging.info('Published message to all Redis targets.')

loop = asyncio.get_event_loop()
manager = QueueManager()
loop.run_until_complete(manager.start())
