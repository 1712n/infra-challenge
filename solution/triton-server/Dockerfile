FROM ghcr.io/els-rd/transformer-deploy:0.4.0

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
COPY ./ $WORKDIR

RUN pip install -U -r requirements.txt
RUN git init && dvc pull
