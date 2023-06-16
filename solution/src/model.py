from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import sys
import torch


log = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(20)
formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)
log.setLevel(20)


class Model():
    def __init__(
        self,
        model_settings,
    ):
        self.model_name = model_settings.values()[0].to_dict()["model_name"]
        self.model_name_short = list(model_settings.keys())[0]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info(f"Model name: {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.label_mapping = model_settings.values()[0].to_dict()["label_mapping"]

    def predict(self, text):

        # Preprocess text
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)

        self.model.to(self.device)

        # Run inference
        output = self.model(**input)
        logits = output.logits
        probabilities = logits.softmax(dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
        predicted_label = self.label_mapping[predicted_class]
        confidence = probabilities[0, predicted_class].item()

        return {
            "score": confidence,
            "label": predicted_label
        }
