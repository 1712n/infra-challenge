FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Install dependencies
COPY requirements.txt .
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base && \
    pip install --no-cache-dir -r requirements.txt

COPY src /opt/app
ENV PYTHONPATH=/opt/app:$PYTHONPATH
WORKDIR /opt/app
ENTRYPOINT ["./start.sh"]
