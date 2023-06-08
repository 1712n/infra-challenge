# server.py
import asyncio
import threading
import logging
from typing import Union, List, Dict
from fastapi import FastAPI
from worker import Worker
from responder import Responder
from main_handler import process_data
from config import config  # Import the configuration

app = FastAPI()

# Configure logging
logger = logging.getLogger('server')
logger.setLevel(logging.DEBUG)

futures_dict = {}
worker_queues = [asyncio.Queue() for _ in range(5)]  # Create 5 worker queues
result_queue = asyncio.Queue()  # Create a single result queue

def start_worker(worker_instance):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(worker_instance.start())
    loop.close()

def start_responder(responder_instance):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(responder_instance.start())
    loop.close()

@app.on_event("startup")
async def startup_event():
    # Start the workers in separate threads
    worker_threads = []
    worker_events = []
    for i, queue in enumerate(worker_queues):
        logging.info(f'Starting worker with queue {id(queue)}.')
        worker_config = config["workers"][i]  # Get the configuration for this worker
        worker_event = threading.Event()
        worker_instance = Worker(worker_config, queue, result_queue, worker_event, logger)
        worker_thread = threading.Thread(target=start_worker, args=(worker_instance,))
        worker_thread.start()
        worker_threads.append(worker_thread)
        worker_events.append(worker_event)

    # Wait for each worker to fully load before starting the next one
    for worker_event in worker_events:
        worker_event.wait()

    # Start the Responder in a separate thread
    responder_instance = Responder(futures_dict, result_queue, logger)
    responder_thread = threading.Thread(target=start_responder, args=(responder_instance,))
    responder_thread.start()

    logging.info('Server started, running result listener')

@app.post("/process")
async def process_endpoint(data: Union[str, List[Dict[str, str]], Dict[str, str]]):
    return await process_data(data, worker_queues, futures_dict, logger)
