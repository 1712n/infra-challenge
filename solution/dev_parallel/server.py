import tornado.ioloop
import tornado.web
import tornado.gen
import json
import redis
import uuid
import logging
from responder import result_listener
from os import environ

# Configure logging
logging.basicConfig(level=logging.DEBUG)

REDIS_PASSWORD = environ.get("REDIS_PASSWORD")
REDIS_HOST = environ.get("REDIS_HOST")

r = redis.Redis(host=REDIS_HOST, password=REDIS_PASSWORD, port=6379, db=0)

futures_dict = {}

class MainHandler(tornado.web.RequestHandler):
    async def post(self):
        data = json.loads(self.request.body)
        correlation_id = str(uuid.uuid4())

        logging.info(f'Received data: {data}. Assigned correlation_id: {correlation_id}.')

        # Назначение задачи для воркера
        message = {
            "correlation_id": correlation_id,
            "data": data,
        }
        r.rpush('dispatcher', json.dumps(message))
        logging.info(f'Pushed to Redis: {message}')

        # Создание Future и его сохранение в словаре
        future = tornado.gen.Future()
        futures_dict[correlation_id] = future
        logging.info(f'Future created for correlation_id: {correlation_id}.')

        # Ожидание результата и его запись в ответ
        result = await future
        logging.info(f'Received result for correlation_id: {correlation_id}. Result: {result}')
        self.write(json.dumps(result))

def make_app():
    return tornado.web.Application([
        (r"/process", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8000, address='0.0.0.0')
    logging.info('Server started.')
    tornado.ioloop.IOLoop.current().spawn_callback(result_listener, futures_dict)
    logging.info('Server started, running result listener')
    tornado.ioloop.IOLoop.current().start()
