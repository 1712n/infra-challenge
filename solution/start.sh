#!/bin/sh
cd triton-server && git init && dvc pull && \
tritonserver --model-repository=/src/triton-server/models & \
	uvicorn app:app --app-dir="/src/solution/app/" --host 0.0.0.0 --port 8080 --workers 5
