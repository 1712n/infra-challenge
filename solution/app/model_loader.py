from transformers import AutoModelForSequenceClassification, AutoTokenizer
import redis
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class ModelLoader:
    def __init__(self):
        self.redis_client = redis.Redis(host="redis", port=6379, db=0)
        self.models = {}
        self.executor = ThreadPoolExecutor()

    async def load_model(self, model_name, model_url):
        # Check if model is cached in Redis
        if self.redis_client.exists(model_name):
            model_bytes = self.redis_client.get(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_bytes)
            tokenizer = AutoTokenizer.from_pretrained(model_bytes)
            self.models[model_name] = {"model": model, "tokenizer": tokenizer}
        else:
            # Load model from Hugging Face model hub
            model = AutoModelForSequenceClassification.from_pretrained(model_url)
            tokenizer = AutoTokenizer.from_pretrained(model_url)
            self.models[model_name] = {"model": model, "tokenizer": tokenizer}

            # Cache the model in Redis
            model_bytes = model_url.encode("utf-8")
            self.redis_client.set(model_name, model_bytes)

    async def load_models(self):
        model_urls = {
            "cardiffnlp": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "ivanlau": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
            "svalabs": "svalabs/twitter-xlm-roberta-crypto-spam",
            "EIStakovskii": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
            "jy46604790": "jy46604790/Fake-News-Bert-Detect"
        }

        for model_name, model_url in model_urls.items():
            await self.load_model(model_name, model_url)

    def get_models(self):
        return self.models

    def process_request(self, text):
        results = {}

        def process_model(model_name, model, tokenizer):
            # Perform inference for a single model
            inputs = tokenizer.encode_plus(text, padding="longest", truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits.detach().squeeze().tolist()
            results[model_name] = {"score": logits[0], "label": logits[1]}

        # Perform inference concurrently for all models
        threads = []
        for model_name, model_data in self.models.items():
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            thread = threading.Thread(target=process_model, args=(model_name, model, tokenizer))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return results
