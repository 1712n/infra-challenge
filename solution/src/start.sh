#!/bin/bash
set -eo pipefail

#Activate conda
source /opt/conda/etc/profile.d/conda.sh && conda activate base

gunicorn main:app --workers=2 --bind=:8080 --worker-class=uvicorn.workers.UvicornWorker --timeout 300
