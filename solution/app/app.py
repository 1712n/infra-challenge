from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from model_loader import ModelLoader
from inference import InferenceEngine
from response_builder import ResponseBuilder
from error_handler import ErrorHandler

app = Flask(__name__)
model_loader = ModelLoader()
inference_engine = InferenceEngine(model_loader)
response_builder = ResponseBuilder()
error_handler = ErrorHandler()

executor = ThreadPoolExecutor()


@app.route("/process", methods=["POST"])
def process_request():
    if not request.is_json or "text" not in request.json:
        error_response = error_handler.handle_invalid_request()
        return jsonify(error_response), 400

    text = request.json["text"]
    try:
        results = inference_engine.process_request(text, executor)
        response = response_builder.build_response(results)
        return jsonify(response), 200
    except Exception:
        error_response = error_handler.handle_model_loading_failure()
        return jsonify(error_response), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
