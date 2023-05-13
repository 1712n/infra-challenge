FROM huggingface/transformers-pytorch-gpu
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"

COPY requirements.txt $WORKDIR

RUN apt-get update && apt upgrade -y && \
		apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx ffmpeg && \
		pip install -U pip setuptools && \
		pip install -U --no-cache-dir -r requirements.txt

COPY . $WORKDIR

# ENTRYPOINT [ "python3", "app.py" ]
ENTRYPOINT [ "bash" ]
