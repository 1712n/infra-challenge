# syntax = docker/dockerfile:1.4

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    git-lfs \
    jq && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt 
COPY app.py /code
COPY cmd.sh .

ENV TRANSFORMERS_CACHE=/models \
    TOKENIZERS_PARALLELISM=false

CMD ["./cmd.sh"] 