class ErrorHandler:
    def handle_invalid_request(self):
        error_response = {
            "error": "Invalid request",
            "message": "Please provide a valid JSON request with the 'text' field."
        }
        return error_response

    def handle_model_loading_failure(self):
        error_response = {
            "error": "Model loading failure",
            "message": "Failed to load one or more NLP models."
        }
        return error_response
