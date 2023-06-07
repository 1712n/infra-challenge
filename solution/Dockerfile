FROM ghcr.io/els-rd/transformer-deploy:0.4.0

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"

COPY requirements.txt $WORKDIR

RUN pip install -U --no-cache-dir -r requirements.txt

COPY . $WORKDIR

ENTRYPOINT [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1" ]
