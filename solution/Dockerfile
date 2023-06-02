FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as builder

FROM python:3.11-slim

LABEL authors="sergeyrudenko"

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip

RUN pip install torch
RUN pip install xformers

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


#
COPY ./main.py /code/

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--log-level", "critical"]