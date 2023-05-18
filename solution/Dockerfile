FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements/requirements.txt ./requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && python3 -m pip install -r requirements.txt

ENV PYTHONPATH "$PYTHONPATH:/app/src"

COPY scripts/start.sh ./start.sh
RUN chmod +x ./start.sh

COPY src/ src/

ENTRYPOINT ["./start.sh"]
EXPOSE 8000
