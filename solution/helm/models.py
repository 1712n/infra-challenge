import asyncio
from transformers import pipeline

# Define the models and their corresponding tasks
models = {
    "cardiffnlp": ("sentiment-analysis", "cardiffnlp/twitter-xlm-roberta-base-sentiment"),
    "ivanlau": ("text-classification", "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base"),
    "svalabs": ("text-classification", "svalabs/twitter-xlm-roberta-crypto-spam"),
    "EIStakovskii": ("text-classification", "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus"),
    "jy46604790": ("text-classification", "jy46604790/Fake-News-Bert-Detect")
}

# Initialize the models
initialized_models = {}

async def initialize_models():
    try:
        for model_name, (task, model) in models.items():
            initialized_models[model_name] = pipeline(task, model=model)
    except Exception as e:
        # Handle any exceptions and raise an error
        raise RuntimeError(f"Failed to initialize models: {str(e)}")
