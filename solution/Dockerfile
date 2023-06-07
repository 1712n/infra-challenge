# sudo apt  install docker-compose

FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

COPY requirements.txt .

RUN  apt-get update \
  && apt-get install -y curl tzdata wget unzip sudo software-properties-common
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN apt-get install -y git-all git-lfs \
  && git lfs install
RUN pip install --upgrade pip \
  && pip uninstall fastapi dataclasses aiofiles pydantic \
  && pip install --no-cache-dir -U -r requirements.txt

WORKDIR /code/
ENV PYTHONPATH=/code/
COPY ./src /code/src
COPY ./config /code/config
COPY ./helm /code/helm

RUN python src/utils.py

EXPOSE 8000