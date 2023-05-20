from collections import defaultdict

from transformers import pipeline

from .conf import models_dict

models = {
    model_name: pipeline(
        task=value["task"],
        model=value["model_path"],
        tokenizer=value["model_path"],
        device="cuda:0",
    )
    for model_name, value in models_dict.items()
}


def get_all_predictions(text: str) -> dict:
    global models
    result = defaultdict(dict)
    for model_name, model in models.items():
        result[model_name] = model(text)[0]
    return result
