#!/bin/sh
python3 models.py && \
uvicorn main:app --workers=5 --worker-class=uvicorn.workers.UvicornWorker
