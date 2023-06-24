#!/bin/bash

set -e
MODELS_DIR='/models'

if [ -d "$MODELS_DIR" ]
then
    echo 'cloning repos...'
    git clone https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment /models/twitter-xlm-roberta-base-sentiment
    git clone https://huggingface.co/ivanlau/language-detection-fine-tuned-on-xlm-roberta-base /models/language-detection-fine-tuned-on-xlm-roberta-base
    cd /models/language-detection-fine-tuned-on-xlm-roberta-base
    tmp=$(mktemp) && jq --arg skipLibCheck true ' .label2id |= with_entries(.value |= tonumber) ' config.json > "$tmp" && mv "$tmp" config.json
    git clone https://huggingface.co/svalabs/twitter-xlm-roberta-crypto-spam /models/twitter-xlm-roberta-crypto-spam
    git clone https://huggingface.co/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus /models/xlm_roberta_base_multilingual_toxicity_classifier_plus
    cd /models/xlm_roberta_base_multilingual_toxicity_classifier_plus
    tmp=$(mktemp) && jq --arg skipLibCheck true '. += { id2label : { "0": "LABEL_0", "1": "LABEL_1" } }' config.json > "$tmp" && mv "$tmp" config.json
    git clone https://huggingface.co/jy46604790/Fake-News-Bert-Detect /models/Fake-News-Bert-Detect
    cd /models/Fake-News-Bert-Detect
    tmp=$(mktemp) && jq --arg skipLibCheck true '. += { id2label : { "0": "LABEL_0", "1": "LABEL_1" } }' config.json > "$tmp" && mv "$tmp" config.json
    cd
    echo 'done'

    model_files=(
        "/models/twitter-xlm-roberta-base-sentiment/"
        "/models/language-detection-fine-tuned-on-xlm-roberta-base/"
        "/models/twitter-xlm-roberta-crypto-spam/"
        "/models/xlm_roberta_base_multilingual_toxicity_classifier_plus/"
        "/models/Fake-News-Bert-Detect/"
    )

    for file_path in "${model_files[@]}"
    do
        rm -f "$file_path/tf_model.h5"
        optimum-cli export onnx -m "$file_path" --device cuda --framework pt --optimize O4 --task text-classification "$file_path"
    done
else
	echo "directory $MODELS_DIR is not mounted, exiting"
    exit
fi

echo 'all done, starting the server'

cd /code
uvicorn app:app --host 0.0.0.0 --port 8080