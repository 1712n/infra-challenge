import asyncio

from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


MODELS = {
    "cardiffnlp": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "ivanlau": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
    "svalabs": "svalabs/twitter-xlm-roberta-crypto-spam",
    "EIStakovskii": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
    "jy46604790": "jy46604790/Fake-News-Bert-Detect",
}

class Pipelines:

    pipelines = {}

    def register_model(self, model: str, model_path: str, tokenizer_path: str):
        try:
            model_instance = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer_path)
            pipe = pipeline(
                "text-classification",
                model=model_instance,
                tokenizer=tokenizer_instance,
                device=0
            )
        except Exception as e:
            print("Couldn't register a model", model)
            raise(e)

        self.pipelines[model] = pipe


    async def query_model(self, model: str, text: str) -> str:
        if model not in self.pipelines:
            return None

        return model, self.pipelines[model](text)[0]


    async def process(self, text: str):
        futures = []
        for model in MODELS:
            futures.append(
                asyncio.ensure_future(self.query_model(model, text))
            )
        results = await asyncio.gather(*futures)

        output = {
            model: result for model, result in results
        }
        if "cardiffnlp" in output:
            output["cardiffnlp"]["label"] = output["cardiffnlp"]["label"].upper()

        return output


def get_pipelines() -> Pipelines:
    pipelines = Pipelines()
    for model, path in MODELS.items():
        pipelines.register_model(
            model=model,
            model_path=path,
            tokenizer_path=path,
        )
    return pipelines

