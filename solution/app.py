# Import required libraries
from fastapi import FastAPI
from fastapi import Request, Body 
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline
from transformers import AutoTokenizer
import onnxruntime

app = FastAPI()

# Load the trained models
print("Model Loading Started...")

# CardiffNLP Model
cardiffnlp_MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment" 
cardiffnlp_ort_model = ORTModelForSequenceClassification.from_pretrained(
    cardiffnlp_MODEL,
    export=True,
    provider="CUDAExecutionProvider", 
    use_io_binding=True
)
cardiffnlp_tokenizer = AutoTokenizer.from_pretrained(cardiffnlp_MODEL) 
print("cardiffnlp Model Converted and Loaded!")

# ivanlau Model
ivanlau_MODEL = f"ivanlau/language-detection-fine-tuned-on-xlm-roberta-base" 
ivanlau_ort_model = ORTModelForSequenceClassification.from_pretrained(
   ivanlau_MODEL,
    export=True,
    provider="CUDAExecutionProvider", 
    use_io_binding=True
)
ivanlau_tokenizer = AutoTokenizer.from_pretrained(ivanlau_MODEL)
print("ivanlau Model Converted and Loaded!")

# svalabs Model
svalabs_MODEL = f"svalabs/twitter-xlm-roberta-crypto-spam" 
svalabs_ort_model = ORTModelForSequenceClassification.from_pretrained(
   svalabs_MODEL,
    export=True,
    provider="CUDAExecutionProvider", 
    use_io_binding=True
)
svalabs_tokenizer = AutoTokenizer.from_pretrained(svalabs_MODEL) 
print("svalabs Model Converted and Loaded!")

# EIStakovskii Model
EIStakovskii_MODEL = f"EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus" 
EIStakovskii_ort_model = ORTModelForSequenceClassification.from_pretrained(
   EIStakovskii_MODEL,
    export=True,
    provider="CUDAExecutionProvider", 
    use_io_binding=True
)
EIStakovskii_tokenizer = AutoTokenizer.from_pretrained(EIStakovskii_MODEL)
print("EIStakovskii Model Converted and Loaded!")

# jy46604790 Model
jy46604790_MODEL = f"jy46604790/Fake-News-Bert-Detect" 
jy46604790_ort_model = ORTModelForSequenceClassification.from_pretrained(
   jy46604790_MODEL,
    export=True,
    provider="CUDAExecutionProvider", 
    use_io_binding=True
)
jy46604790_tokenizer = AutoTokenizer.from_pretrained(jy46604790_MODEL)
print("jy46604790 Model Converted and Loaded!")



# Define a default route
@app.get('/')
async def home():
    html = (
        'Fast Inference Mlops Challenge'
    )
    return html.format(format)


def preprocess(text: str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)



# Define a function for each of the trained models to preprocess the input and return the prediction
def cardiffnlp_precessing(text: str) -> dict:
    pipe = pipeline(task="text-classification", model=cardiffnlp_ort_model, tokenizer=cardiffnlp_tokenizer,device=0)
    result = pipe(text) 
    result[0]['label'] = result[0]['label'].upper() 
    
    result = {
    "score": result[0]['score'],
    "label": result[0]['label']
    } 
    return result 


def ivanlau_precessing(text: str) -> dict:
    pipe = pipeline(task="text-classification", model=ivanlau_ort_model, tokenizer=ivanlau_tokenizer,device=0)
    result = pipe(text) 
    result = {
    "score": result[0]['score'],
    "label": result[0]['label']
    } 
    return result


def svalabs_precessing(text: str) -> dict:
    pipe = pipeline(task="text-classification", model=svalabs_ort_model, tokenizer=svalabs_tokenizer,device=0)
    result = pipe(text) 
    result = {
    "score": result[0]['score'],
    "label": result[0]['label']
    } 
    return result


def EIStakovskii_precessing(text: str) -> dict:
    pipe = pipeline(task="text-classification", model=EIStakovskii_ort_model, tokenizer=EIStakovskii_tokenizer,device=0)
    result = pipe(text) 
    result = {
    "score": result[0]['score'],
    "label": result[0]['label']
    } 
    return result

def jy46604790_precessing(text: str) -> dict:
    pipe = pipeline(task="text-classification", model=jy46604790_ort_model, tokenizer=jy46604790_tokenizer,device=0)
    result = pipe(text) 
    result = {
    "score": result[0]['score'],
    "label": result[0]['label']
    } 
    return result


# Define a process route
@app.post("/process")
async def predict(request: Request): 
    # start = time.time()
    text = preprocess((await request.body()).decode())
    results_cardiffnlp =  cardiffnlp_precessing(text)
    results_ivanlau = ivanlau_precessing(text)
    results_svalabs = svalabs_precessing(text)
    results_EIStakovskii = EIStakovskii_precessing(text)
    results_jy46604790 = jy46604790_precessing(text)
    # end = time.time()
    # total_time = end - start
    # print(f"Total inference time with preprocessing: {total_time}")
    result = {
        "cardiffnlp": results_cardiffnlp,
        "ivanlau":results_ivanlau,
        "svalabs":results_svalabs,
        "EIStakovskii":results_EIStakovskii,
        "jy46604790":results_jy46604790,
        }
    return result  