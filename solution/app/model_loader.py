from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ModelLoader:
    def __init__(self):
        self.models = {}

    def load_models(self):
        model_urls = {
            "cardiffnlp": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "ivanlau": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
            "svalabs": "svalabs/twitter-xlm-roberta-crypto-spam",
            "EIStakovskii": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
            "jy46604790": "jy46604790/Fake-News-Bert-Detect"
        }

        for model_name, model_url in model_urls.items():
            model = AutoModelForSequenceClassification.from_pretrained(model_url)
            tokenizer = AutoTokenizer.from_pretrained(model_url)
            self.models[model_name] = {"model": model, "tokenizer": tokenizer}

    def get_models(self):
        return self.models
