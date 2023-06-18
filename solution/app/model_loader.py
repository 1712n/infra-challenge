from transformers import AutoModelForSequenceClassification, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import threading

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.executor = ThreadPoolExecutor(max_workers=5)

    def load_models(self):
        model_urls = {
            "cardiffnlp": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "ivanlau": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
            "svalabs": "svalabs/twitter-xlm-roberta-crypto-spam",
            "EIStakovskii": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
            "jy46604790": "jy46604790/Fake-News-Bert-Detect"
        }

        def load_model(model_name, model_url):
            model = AutoModelForSequenceClassification.from_pretrained(model_url)
            tokenizer = AutoTokenizer.from_pretrained(model_url)
            self.models[model_name] = {"model": model, "tokenizer": tokenizer}

        # Load models asynchronously using ThreadPoolExecutor
        futures = []
        for model_name, model_url in model_urls.items():
            future = self.executor.submit(load_model, model_name, model_url)
            futures.append(future)

        # Wait for all model loading tasks to complete
        threading.current_thread().set_name("Main")
        for future in futures:
            future.result()

    def get_models(self):
        return self.models
