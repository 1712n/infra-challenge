import concurrent.futures
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import threading


class InferenceEngine:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def process_request(self, text, executor):
        models = self.model_loader.get_models()
        futures = []

        for model_name, model_data in models.items():
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            future = executor.submit(self.run_inference, model_name, model, tokenizer, text)
            futures.append(future)

        results = {}
        for future in concurrent.futures.as_completed(futures):
            model_name, output = future.result()
            results[model_name] = output

        return results

    def run_inference(self, model_name, model, tokenizer, text):
        with model.as_default():
            inputs = tokenizer.encode_plus(text, padding="longest", truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits.detach().squeeze().tolist()
            score, label = logits[0], logits[1]
        return model_name, {"score": score, "label": label}
