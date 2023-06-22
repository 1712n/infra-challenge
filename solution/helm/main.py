import asyncio
import time
import numpy as np
from transformers import AutoConfig , AutoTokenizer
from onnxruntime.transformers.io_binding_helper import IOBindingHelper
from onnxruntime import InferenceSession, GraphOptimizationLevel
import torch
from fastapi import FastAPI, Request
from cachetools import TTLCache

app = FastAPI()

# Load the NLP models
models = {
    "cardiffnlp": {
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "tokenizer": AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment"),
        "model": InferenceSession("models/optimized_cardiffnlp/model.onnx",providers=['TensorrtExecutionProvider', "CUDAExecutionProvider"]),
        "id2label": AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment").id2label
    },
    "ivanlau": {
        "model_name": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
        "tokenizer": AutoTokenizer.from_pretrained("ivanlau/language-detection-fine-tuned-on-xlm-roberta-base"),
        "model": InferenceSession("models/optimized_ivanlau/model.onnx",providers=['TensorrtExecutionProvider', "CUDAExecutionProvider"]),
        "id2label": AutoConfig.from_pretrained("ivanlau/language-detection-fine-tuned-on-xlm-roberta-base").id2label
    },
    "svalabs": {
        "model_name": "svalabs/twitter-xlm-roberta-crypto-spam",
        "tokenizer": AutoTokenizer.from_pretrained("svalabs/twitter-xlm-roberta-crypto-spam"),
        "model": InferenceSession("models/optimized_svalabs/model.onnx",providers=['TensorrtExecutionProvider', "CUDAExecutionProvider"]),
        "id2label": AutoConfig.from_pretrained("svalabs/twitter-xlm-roberta-crypto-spam").id2label
    },
    "EIStakovskii": {
        "model_name": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
        "tokenizer": AutoTokenizer.from_pretrained("EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus"),
        "model": InferenceSession("models/optimized_EIStakovskii/model.onnx",providers=['TensorrtExecutionProvider', "CUDAExecutionProvider"]),
        "id2label": AutoConfig.from_pretrained("EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus").id2label
    },
    "jy46604790": {
        "model_name": "jy46604790/Fake-News-Bert-Detect",
        "tokenizer": AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect"),
        "model": InferenceSession("models/optimized_jy46604790/model.onnx",providers=['TensorrtExecutionProvider', "CUDAExecutionProvider"]),
        "id2label": AutoConfig.from_pretrained("jy46604790/Fake-News-Bert-Detect").id2label
    },
}

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_io_binding(
    ort_session, input_dict, output_buffers, output_shapes
):
    """
    Prepare the input/output binding for the provided ONNX Runtime session. 

    :param ort_session: The ONNX Runtime session to bind inputs/outputs
    :type ort_session: onnxruntime.InferenceSession

    :param input_dict: A dictionary containing the input names and values
    :type input_dict: dict[str, numpy.ndarray]

    :param output_buffers: A dictionary containing the output buffers to store the output values
    :type output_buffers: dict[str, numpy.ndarray]

    :param output_shapes: A dictionary containing the output shapes
    :type output_shapes: dict[str, List[int]]

    :return: The IO binding for the provided ONNX Runtime session
    :rtype: onnxruntime.OrtDevice
    """
    ort_session.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    io_binding = ort_session.io_binding()

    # Bind inputs
    for name, input_val in input_dict.items():
        if input_val is not None:
            input_val = torch.from_numpy(input_val)
            io_binding.bind_input(
                    name,
                    input_val.device.type,
                    0,
                    np.int64,
                    input_val.size(),
                    input_val.data_ptr(),
            )

    # Bind outputs
    for output in ort_session.get_outputs():
        output_name = output.name
        output_buffer = output_buffers[output_name]
        io_binding.bind_output(
            output_name,
            "cuda",
            0,
            np.float32,
            output_shapes[output_name],
            output_buffer.data_ptr(),
        )

    return io_binding

async def model_inference(model, labels, inputs):
    """
    Asynchronously performs inference on a PyTorch model using the provided inputs.

    :param model: The PyTorch model to perform inference on.
    :type model: torch.nn.Module
    :param labels: A list of labels corresponding to the model's output classes.
    :type labels: List[str]
    :param inputs: The inputs to be passed to the model for inference.
    :type inputs: torch.Tensor
    :return: A dictionary containing the highest scoring label and its corresponding score.
    :rtype: Dict[str, Union[str, float]]
    """
    output_buffers = {
    "logits": torch.empty(
        (model.get_outputs()[0].shape[1],), dtype=torch.float32, device="cuda"
    ),
}
    output_shapes = {
    "logits": [1, model.get_outputs()[0].shape[1]],
}
    io_binding = prepare_io_binding(
    model,
    inputs,
    output_buffers,
    output_shapes,
)

    model.run_with_iobinding(io_binding)
    outputs = IOBindingHelper.get_outputs_from_io_binding_buffer(
        model, output_buffers=output_buffers, output_shapes=output_shapes
    )
    outputs = torch.from_numpy(outputs[0])
    scores = torch.nn.functional.softmax(outputs)[0]
    max_i = scores.argmax().item()
    return {"score": scores[max_i].item(), "label": labels[max_i]}
    

@app.post("/process")
async def process(request: Request):
    # initiate timer
    start_time = time.time()
    text = (await request.body()).decode("utf-8")
    results = TTLCache(maxsize=5000, ttl=300)
    tasks = []

    for model_name, model_data in models.items():
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]

        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        inputs = {key: np.array(val, dtype=np.int64) for key, val in inputs.items()}
        tasks.append(asyncio.create_task(model_inference(model, model_data["id2label"], inputs)))
    results = await asyncio.gather(*tasks)
    results_dict = dict(zip(list(models.keys()), results))
    end_time = time.time()
    print(end_time - start_time)

    return results_dict

if __name__ == "__main__":
    app.run(host="0.0.0.0")
