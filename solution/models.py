import os
from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime.transformers.io_binding_helper import IOBindingHelper
from onnxruntime.transformers.optimizer import optimize_model
import torch

models = {
    "cardiffnlp": {
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "tokenizer": None,
        "model": None,
    },
    "ivanlau": {
        "model_name": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
        "tokenizer": None,
        "model": None,
    },
    "svalabs": {
        "model_name": "svalabs/twitter-xlm-roberta-crypto-spam",
        "tokenizer": None,
        "model": None,
    },
    "EIStakovskii": {
        "model_name": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
        "tokenizer": None,
        "model": None,
    },
    "jy46604790": {
        "model_name": "jy46604790/Fake-News-Bert-Detect",
        "tokenizer": None,
        "model": None,
    },
}

def download_model(model_name):
    optimized_onnx_path = f"optimized_{model_name}"
    model_data = models[model_name]
    model_data["optimized_onnx_path"] = os.path.join("models", optimized_onnx_path)
    model_dir = os.path.join("models", model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = ORTModelForSequenceClassification.from_pretrained(model_data["model_name"], export=True, use_io_binding=False).to(device)
    model.save_pretrained(str("models/" + optimized_onnx_path))
    model = optimize_model(
    input=str("models/" + optimized_onnx_path + "/model.onnx"),
    model_type="bert",
    use_gpu=True
)
    model.save_model_to_file(str("models/" + optimized_onnx_path + "/model.onnx"))

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name in models:
    torch.cuda.empty_cache()
    download_model(model_name)
    print(model_name)