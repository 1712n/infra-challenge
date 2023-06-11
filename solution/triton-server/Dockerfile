FROM ghcr.io/els-rd/transformer-deploy:0.4.0

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
COPY ./ $WORKDIR

RUN pip install -U --no-cache-dir -r requirements.txt

