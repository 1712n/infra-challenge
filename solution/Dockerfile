FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as build

WORKDIR /app

COPY requirements/ ./

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN python3 -m pip install -r requirements.txt

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as prod

ENV PYTHONPATH "$PYTHONPATH:/app/src"
ENV PATH="/app/venv/bin:$PATH"

WORKDIR /app

COPY --from=build /app/venv ./venv

COPY src/ src/
COPY scripts/start.sh ./start.sh

RUN chmod +x ./start.sh

ENTRYPOINT ["./start.sh"]
EXPOSE 8000
