# Import required libraries
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig  
import numpy as np  
from scipy.special import softmax 
from fastapi import Request, Body
# Initialize a FastAPI app instance
app = FastAPI()

# Define a default route
@app.get('/')
async def home():
    html = (
        'Mlops Challenge'
    )
    return html.format(format)


# Load the trained models
print("Model Loading Started...")

# CardiffNLP Model
cardiffnlp_MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
cardiffnlp_tokenizer = AutoTokenizer.from_pretrained(cardiffnlp_MODEL) 
cardiffnlp_model = AutoModelForSequenceClassification.from_pretrained(cardiffnlp_MODEL)
print("cardiffnlp Model Loaded!")

# ivanlau Model
ivanlau_MODEL = f"ivanlau/language-detection-fine-tuned-on-xlm-roberta-base"
ivanlau_tokenizer = AutoTokenizer.from_pretrained(ivanlau_MODEL)
ivanlau_config = AutoConfig.from_pretrained(ivanlau_MODEL)
ivanlau_model = AutoModelForSequenceClassification.from_pretrained(ivanlau_MODEL)
print("ivanlau Model Loaded!")

# svalabs Model
svalabs_MODEL = f"svalabs/twitter-xlm-roberta-crypto-spam"
svalabs_tokenizer = AutoTokenizer.from_pretrained(svalabs_MODEL) 
svalabs_model = AutoModelForSequenceClassification.from_pretrained(svalabs_MODEL)
print("svalabs Model Loaded!")

# EIStakovskii Model
EIStakovskii_MODEL = f"EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus"
EIStakovskii_tokenizer = AutoTokenizer.from_pretrained(EIStakovskii_MODEL)
EIStakovskii_model = AutoModelForSequenceClassification.from_pretrained(EIStakovskii_MODEL)
print("EIStakovskii Model Loaded!")

# jy46604790 Model
jy46604790_MODEL = f"jy46604790/Fake-News-Bert-Detect"
jy46604790_tokenizer = AutoTokenizer.from_pretrained(jy46604790_MODEL)
jy46604790_model = AutoModelForSequenceClassification.from_pretrained(jy46604790_MODEL)
print("jy46604790 Model Loaded!")

 
def preprocess(text: str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# Define a function for each of the trained models to preprocess the input and return the prediction
def cardiffnlp_precessing(text: str) -> dict:
    encoded_input = cardiffnlp_tokenizer(text, return_tensors='pt')
    outputs = cardiffnlp_model(**encoded_input)
    outputs = outputs.logits.detach().numpy()
    
    scores = softmax(outputs)[0] 
    
    max_index = scores.argmax() 

    id2label = ["NEGATIVE","NEUTRAL","POSITIVE"]
    
    l = id2label[max_index]
    s = scores[max_index]
    result = {
    "score": float(s),
    "label": l
    } 
    return result

def ivanlau_precessing(text: str) -> dict:
    encoded_input = ivanlau_tokenizer(text, return_tensors='pt')
    outputs = ivanlau_model(**encoded_input)
    outputs = outputs.logits.detach().numpy()
    
    scores = softmax(outputs)[0] 
    
    max_index = scores.argmax() 


    l = ivanlau_config.id2label[max_index]
    s = scores[max_index]
    result = {
            "score": float(s),
            "label": l
        } 
    return result

def svalabs_precessing(text: str) -> dict:
    encoded_input = svalabs_tokenizer(text, return_tensors='pt')
    outputs = svalabs_model(**encoded_input)
    outputs = outputs.logits.detach().numpy()

    scores = softmax(outputs)[0] 
    
    id2label = ["HAM","SPAM"]
    
    max_index = scores.argmax() 
    
    l = id2label[max_index]
    s = scores[max_index]
    result = {
    "score": float(s),
    "label": l
    } 
    return result

def EIStakovskii_precessing(text: str) -> dict:
    encoded_input = EIStakovskii_tokenizer(text, return_tensors='pt')
    outputs = EIStakovskii_model(**encoded_input)
    outputs = outputs.logits.detach().numpy()

    scores = softmax(outputs)[0] 

    max_index = scores.argmax() 
    
    id2label = ["LABEL_0","LABEL_1"]
    
    l = id2label[max_index]
    s = scores[max_index]
    result = {
    "score": float(s),
    "label": l
    } 
    
    return result

def jy46604790_precessing(text: str) -> dict:
    encoded_input = jy46604790_tokenizer(text, return_tensors='pt')
    outputs = jy46604790_model(**encoded_input)
    outputs = outputs.logits.detach().numpy()
    
    scores = softmax(outputs)[0] 
    
    max_index = scores.argmax() 

    
    id2label = ["LABEL_1","LABEL_0"]
    
    l = id2label[max_index]
    s = scores[max_index]
    result = {
    "score": float(s),
    "label": l
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