import asyncio
import logging.config
import os
import torch
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from optimum.onnxruntime import ORTModelForSequenceClassification
from starlette.responses import HTMLResponse
from transformers import AutoTokenizer

# Global variables

g_logger = logging.getLogger(__name__)
g_current_path = os.getcwd() + "/models"

# dict {model_name: local path} todo save to some kind of conf file to be more like framework
G_MODELS = {"cardiffnlp": g_current_path + "/cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "ivanlau": g_current_path + "/ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
            "svalabs": g_current_path + "/svalabs/twitter-xlm-roberta-crypto-spam",
            "EIStakovskii": g_current_path + "/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
            "jy46604790": g_current_path + "/jy46604790/Fake-News-Bert-Detect"}

# dict {model_name: (model, tokenizer)}
g_model_pipelines: dict = {}

# dict returning API data {model_name: {score:xx, label: xx} }
g_data: dict = {}

# if GPU available use it
g_device = "cuda:0" if torch.cuda.is_available() else "cpu"
if g_device == "cpu":
    raise Exception("Не тестируем на cpu, нужен гпу!")

# spinoff the api and set log level to critical to speedup todo set log level in env vars
# app = FastAPI(title='Inference', debug=False)
app = FastAPI(title='Inference')


# Helper functions
def register_models(models_dict: dict[str, str]) -> None:
    """helper function to register locally stored models by populating g_model_pipelines dict"""
    g_logger.warning("Start registering models.")
    for model_name, model_path in models_dict.items():
        try:
            g_logger.warning("Registering: %s" % model_name)
            g_model_pipelines[model_name] = (
                ort.InferenceSession(model_path + "/model.onnx", providers=["CUDAExecutionProvider","CPUExecutionProvider"]), AutoTokenizer.from_pretrained(model_path))
            # model.eval()
        except:
            err = " Error while registering:  %s" % model_name
            raise RuntimeError(err)
    g_logger.warning("Finished registering models.")


@app.get("/load", response_class=HTMLResponse)
def download_models() -> HTMLResponse:
    """helper download models to local folder if not downoaded before"""
    # can be invoked manually forcefully by visiting page at localhost:9000/load
    g_logger.warning("Start downloading models from huggingfaces.")
    for mdl in G_MODELS.values():
        model_name = "/".join(mdl.split("/")[-2:])
        save_path = os.getcwd() + "/" + "/".join(mdl.split("/")[-3:])
        g_logger.warning("Loading: %s" % model_name)

        try:
            model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # model.eval()
            model.save_pretrained(save_directory=save_path)
            tokenizer.save_pretrained(save_directory=save_path)
        except:
            err = " Error while loading:  %s" % model_name
            raise RuntimeError(err)

    g_logger.warning("Finished downloading models from huggingfaces.")
    return """
    <html>
        <head>
            <title>Models downloaded</title>
        </head>
        <body>
            <h1>Models downloaded</h1>
        </body>
    </html>
    """


# Core logic functions
@app.on_event("startup")
async def startup_trigger():
    os.environ['TRANSFORMERS_CACHE'] = os.getcwd() + '/models'
    g_logger.setLevel(logging.WARNING)
    g_logger.warning("Starting app...")
    g_logger.warning("Inference device: %s" % g_device)
    if not os.path.exists("models/"):  # naive way to check whether models need to be downloaded todo
        download_models()
    # register locally downloaded models
    register_models(G_MODELS)

    # hopefully a hair faster
    # torch.set_float32_matmul_precision('medium')

    g_logger.warning("API ready for use.")


async def inference(sentence: str, model_name: str, model: tuple) -> None:
    """perform inference"""

    # encoded_sentence = model[1](sentence, return_tensors="pt")  # .to(g_device)
    # encoded_sentence_onnx = {k: v.cpu().numpy() for k, v in model[1](sentence, return_tensors="pt").items()}  # Convert to CPU numpy arr as ONNX expects numpy

    # outs_onnx = model[0].run(None, encoded_sentence_onnx)

    # Get logits
    # logits = outs_onnx[0]

    # Probs
    probs = torch.nn.functional.softmax(torch.from_numpy(model[0].run(None, {k: v.cpu().numpy() for k, v in model[1](sentence, return_tensors="pt").items()})[0]), dim=-1)

    # Get the index of max prob and prob itself
    pred_index = torch.argmax(probs, dim=-1).item()
    confid_score = probs[0, pred_index].item()

    # id2label for onnx not exist so only manually
    # todo save in some kind conf file or db to be more like framework
    match model_name:
        case "cardiffnlp":
            id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        case "ivanlau":
            id2label = {0: "Arabic", 1: "Basque", 2: "Breton", 3: "Catalan", 4: "Chinese_China", 5: "Chinese_Hongkong",
                        6: "Chinese_Taiwan", 7: "Chuvash", 8: "Czech", 9: "Dhivehi", 10: "Dutch", 11: "English",
                        12: "Esperanto", 13: "Estonian", 14: "French", 15: "Frisian", 16: "Georgian", 17: "German",
                        18: "Greek", 19: "Hakha_Chin", 20: "Indonesian", 21: "Interlingua", 22: "Italian",
                        23: "Japanese", 24: "Kabyle", 25: "Kinyarwanda", 26: "Kyrgyz", 27: "Latvian", 28: "Maltese",
                        29: "Mongolian", 30: "Persian", 31: "Polish", 32: "Portuguese", 33: "Romanian",
                        34: "Romansh_Sursilvan", 35: "Russian", 36: "Sakha", 37: "Slovenian", 38: "Spanish",
                        39: "Swedish", 40: "Tamil", 41: "Tatar", 42: "Turkish", 43: "Ukranian", 44: "Welsh"}
        case "svalabs":
            id2label = {0: "HAM", 1: "SPAM"}
        case "EIStakovskii":
            id2label = {0: "LABEL_0", 1: "LABEL_1"}
        case "jy46604790":
            id2label = {0: "LABEL_0", 1: "LABEL_1"}
        case _:
            err = "Error wrong  model name:  %s" % model_name
            raise RuntimeError(err)

    # Get predicted label
    pred_label = id2label[pred_index]

    g_data[model_name] = {"score": confid_score, "label": pred_label}


@app.post("/")
async def all_infers(sentence: str = Body(...)) -> dict:
    """inference endpoint"""
    if not sentence:  # check whether input request text is empty
        g_logger.warning('Request text is empty')
        raise HTTPException(status_code=400, detail="Request text is empty")

    # create inference tasks per model
    tasks = [inference(sentence, model_name, model) for model_name, model in
             g_model_pipelines.items()]

    # run inference tasks and populate resulting dict
    await asyncio.gather(*tasks)
    return g_data


if __name__ == '__main__':
     uvicorn.run("main:app", port=9000, reload=True, log_level="critical")
