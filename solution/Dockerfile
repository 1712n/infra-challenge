FROM python:3.11-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /app
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app/"

CMD ["python", "main.py"]
