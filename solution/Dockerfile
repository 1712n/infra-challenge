# syntax = docker/dockerfile:1.4

# todo: add checking out to specific revision to avoid unexpected changes to model behaviour?

FROM python:3.10.11-slim-bullseye as models_storage_stage

RUN --mount=type=cache,id=storage-apt,target=/var/cache/apt \
    apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    git-lfs

WORKDIR /models

RUN git clone https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment && \
    git clone https://huggingface.co/ivanlau/language-detection-fine-tuned-on-xlm-roberta-base && \
    git clone https://huggingface.co/svalabs/twitter-xlm-roberta-crypto-spam && \
    git clone https://huggingface.co/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus && \
    git clone https://huggingface.co/jy46604790/Fake-News-Bert-Detect

FROM python:3.10.11-slim-bullseye as app 

WORKDIR /code
RUN mkdir -p /code/models
COPY --from=models_storage_stage /models /code/models

COPY ./requirements.txt /code/requirements.txt
RUN --mount=type=cache,id=infra-challenge-pip,target=/root/.cache \
    pip install -r /code/requirements.txt 

COPY app.py /code

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"] 