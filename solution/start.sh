tritonserver --model-repository=/src/triton-server/models & \
	uvicorn app:app --reload --app-dir="$(pwd)/app/" --host 0.0.0.0 --port 8080 --workers 5
