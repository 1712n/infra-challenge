# Использование PyTorch образа с поддержкой CUDA и cuDNN в качестве базового
FROM pytorch/pytorch:latest
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubi8

# Обновление pip
RUN pip install --upgrade pip

# Set environment variables
ARG REDIS_HOST
ENV REDIS_HOST=$REDIS_HOST
ARG REDIS_PASSWORD
ENV REDIS_PASSWORD=$REDIS_PASSWORD

# Set CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    make \
    gcc \
    procps \
    lsof \
    vim \
    supervisor \
    nginx \
    linux-headers-$(uname -r)

COPY default /etc/nginx/sites-available/
COPY default /etc/nginx/sites-enabled/

WORKDIR /solution

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY /dev_parallel/ .

COPY /gunicorn_conf.py .

# Копируйте файл конфигурации supervisord в контейнер
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Используйте CMD для запуска supervisord
CMD ["/usr/bin/supervisord"]
