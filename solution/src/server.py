from fastapi import FastAPI, Request
import asyncio
import logging
import sys
from config.config import settings
from src.model import Model


log = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(20)
formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)
log.setLevel(20)


app = FastAPI()
app.request_queue = asyncio.Queue()
app.models = {}


# Initiating HF model instances from downloaded models
def init_models(models_dict, models_settings):
    if not bool(models_dict):
        for model in models_settings["models"]:

            log.debug(f"Model in progress: {model}")
            model_instance = Model(model)
            
            # Store model
            models_dict[model_instance.model_name] = model_instance
            log.debug(f"Models dict: {models_dict}")
    return models_dict


# Processing text for inference
async def get_inference_results():
    inference_results = {}
    while True:
        job = await app.request_queue.get()
        log.info(f"Got a job: (size of remaining queue: {app.request_queue.qsize()})")
        
        try:
            # Process the received text
            data = await job.body()
            log.info(f"Text: {data.decode()}")
                
            for model in app.models.items():
            
                model_output = model[1].predict(data.decode())
                log.debug(f"After the model's work: {model_output}")

                # Store inference result
                inference_results[model[1].model_name_short] = model_output
                
            log.info(f"Inference results: {inference_results}")
            return inference_results
            
        except Exception as e:
            log.info(f"Exception occurred: {str(e)}")
            
        finally:
            app.request_queue.task_done()


# Start asyncio background task
@app.on_event("startup")
async def startup():
    app.models = init_models(app.models, settings)
    asyncio.create_task(get_inference_results())


# API endpoint for inference
@app.post("/process")
async def process_request(request: Request):
    await app.request_queue.put(request)
    log.info("Request is received and put into queue")
    inference_results = await get_inference_results()
    await asyncio.sleep(0)
    return inference_results

