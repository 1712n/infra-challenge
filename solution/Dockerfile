# syntax = docker/dockerfile:1.4

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    git-lfs

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt 
COPY app.py /code

ENV TRANSFORMERS_CACHE=/models

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"] 