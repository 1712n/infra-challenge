#!/bin/bash
gunicorn main:app --workers=2 --bind=:9000 --worker-class=uvicorn.workers.UvicornWorker --timeout 900 --log-level critical