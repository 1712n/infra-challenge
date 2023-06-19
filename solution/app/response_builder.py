from collections import defaultdict


class ResponseBuilder:
    def build_response(self, results):
        response = defaultdict(dict)
        for model_name, output in results.items():
            score = output["score"]
            label = output["label"]
            response[model_name]["score"] = score
            response[model_name]["label"] = label
        return dict(response)
