FROM python:3.9.1-slim

# Обновление pip
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    wget \
    make \
    gcc \
    procps \
    lsof \
    vim \
    supervisor  

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY /dev_parallel/ .

# Копируйте файл конфигурации supervisord в контейнер
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Используйте CMD для запуска supervisord
CMD ["/usr/bin/supervisord"]
