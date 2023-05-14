import logging

from fastapi import Depends

from helpers.output_functions import label_upper
from ml_models.init_models import load_models
from models.model_process import ModelPipeline
from services.base_service import BaseService

logger = logging.getLogger()


class MLModelProcess(BaseService):
    async def process_data(
        self,
        payload: str,
    ):
        result = {}
        for model in self.models:
            model_result = model.pipeline(payload)
            if len(model_result) > 0:
                model_result = model_result[0]
            else:
                raise Exception("Model result is empty for model: " + model.model_process.model_name)
            if model.model_process.output_functions is not None:
                for output_function in model.model_process.output_functions:
                    func = globals()[output_function]
                    model_result = func(model_result)
            result[model.model_process.model_name] = model_result

        return result


def get_model_service(
    loaded_models: list[ModelPipeline] = Depends(load_models),
) -> MLModelProcess:
    return MLModelProcess(models=loaded_models)
