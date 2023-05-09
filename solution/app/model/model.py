import aiohttp
import asyncio

models = ["cardiffnlp", "ivanlau", "svalabs", "EIStakovskii", "jy46604790"]
API_URLS = ["https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "https://api-inference.huggingface.co/models/ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
            "https://api-inference.huggingface.co/models/svalabs/twitter-xlm-roberta-crypto-spam",
            "https://api-inference.huggingface.co/models/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
            "https://api-inference.huggingface.co/models/jy46604790/Fake-News-Bert-Detect"]
headers = {"Authorization": "Bearer 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"}

async def query_async(payload, api):
    async with aiohttp.ClientSession() as session:
        async with session.post(api, headers=headers, json=payload) as response:
            return await response.json()

async def predict_pipeline(text):
    tasks = []
    for model, api in zip(models, API_URLS):
        task = asyncio.ensure_future(query_async({
            "inputs": text,
            "options": {"wait_for_model": True}
        }, api))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    result = {}
    for model, output in zip(models, results):
        sorted_output = sorted(output[0], key=lambda d: d['score'], reverse=True)
        result[model] = sorted_output[0]

    return result