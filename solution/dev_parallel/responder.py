import json
import asyncio

class Responder:
    def __init__(self, futures_dict, result_queue, logger):
        self.logger = logger
        self.futures_dict = futures_dict
        self.result_queue = result_queue
        self.tasks = {}

    async def fetch_results(self):
        while True:
            result_string = await self.result_queue.get()
            result = json.loads(result_string)
            correlation_id = result.get('correlation_id')
            worker_result = result.get('result')

            if correlation_id not in self.tasks:
                self.tasks[correlation_id] = [0, {}]

            self.tasks[correlation_id][1].update(worker_result)
            self.tasks[correlation_id][0] += 1

            if self.tasks[correlation_id][0] >= 5:
                asyncio.create_task(self.process_results(correlation_id))

    async def process_results(self, correlation_id):
        aggregated_results = self.tasks[correlation_id][1]
        final_message = aggregated_results
        future = self.futures_dict.pop(correlation_id, None)
        if future is not None:
            future.set_result(final_message)
        del self.tasks[correlation_id]

    async def start(self):
        asyncio.create_task(self.fetch_results())
        while True:
            await asyncio.sleep(0.025)
