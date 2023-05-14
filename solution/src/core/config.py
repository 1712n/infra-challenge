import torch
from pydantic import BaseSettings

from models.model_process import ModelProcess


class Settings(BaseSettings):
    project_name: str = "models_api"
    device: int = 0 if torch.cuda.is_available() else -1
    MODELS: list[ModelProcess] = [
        {
            "model_name": "cardiffnlp",
            "model_path": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "output_functions": ["label_upper"],
        },
        {
            "model_name": "jy46604790",
            "model_path": "jy46604790/Fake-News-Bert-Detect",
        },
        {
            "model_name": "ivanlau",
            "model_path": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
        },
        {
            "model_name": "svalabs",
            "model_path": "svalabs/twitter-xlm-roberta-crypto-spam",
        },
        {
            "model_name": "EIStakovskii",
            "model_path": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
        },
    ]

    class Config:
        case_sensitive = False
        env_file = "models_api.env"
        env_file_encoding = "utf-8"


settings = Settings()
