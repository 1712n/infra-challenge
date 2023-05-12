use std::collections::{HashMap};
// use std::path::PathBuf;

use bytes::Bytes;
// use rust_bert::pipelines::sequence_classification::SequenceClassificationConfig;
// use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
// use rust_bert::resources::LocalResource;
use serde::{Deserialize, Serialize};
use serde_json::{from_str};
use warp::Filter;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelResult {
    pub score: f64,
    pub label: String,
}

// let config = SequenceClassificationConfig::new(ModelType::Roberta,
//    LocalResource { local_path: PathBuf::from("/models/twitter-xlm-roberta-base-sentiment/") };
//    LocalResource { local_path: PathBuf::from("/models/twitter-xlm-roberta-base-sentiment/sentencepiece.bpe.model") };
//    LocalResource { local_path: PathBuf::from("/models/twitter-xlm-roberta-base-sentiment/config.json") };
//    None,
//    true,
//    None,
//    None,
// );

// let config = SequenceClassificationConfig::new(ModelType::Roberta,
//    LocalResource { local_path: PathBuf::from("/models/language-detection-fine-tuned-on-xlm-roberta-base/") };
//    LocalResource { local_path: PathBuf::from("/models/language-detection-fine-tuned-on-xlm-roberta-base/sentencepiece.bpe.model") };
//    LocalResource { local_path: PathBuf::from("/models/language-detection-fine-tuned-on-xlm-roberta-base/config.json") };
//    None,
//    true,
//    None,
//    None,
// );

// let config = SequenceClassificationConfig::new(ModelType::Roberta,
//    LocalResource { local_path: PathBuf::from("/models/twitter-xlm-roberta-crypto-spam/") };
//    LocalResource { local_path: PathBuf::from("/models/twitter-xlm-roberta-crypto-spam/sentencepiece.bpe.model") };
//    LocalResource { local_path: PathBuf::from("/models/twitter-xlm-roberta-crypto-spam/config.json") };
//    None,
//    true,
//    None,
//    None,
// );

// let config = SequenceClassificationConfig::new(ModelType::Roberta,
//    LocalResource { local_path: PathBuf::from("/models/xlm_roberta_base_multilingual_toxicity_classifier_plus/") };
//    LocalResource { local_path: PathBuf::from("/models/xlm_roberta_base_multilingual_toxicity_classifier_plus/sentencepiece.bpe.model") };
//    LocalResource { local_path: PathBuf::from("/models/xlm_roberta_base_multilingual_toxicity_classifier_plus/config.json") };
//    None,
//    true,
//    None,
//    None,
// );

// let config = SequenceClassificationConfig::new(ModelType::Roberta,
//    LocalResource { local_path: PathBuf::from("/models/Fake-News-Bert-Detect/") };
//    LocalResource { local_path: PathBuf::from("/models/Fake-News-Bert-Detect/vocab.json") };
//    LocalResource { local_path: PathBuf::from("/models/Fake-News-Bert-Detect/config.json") };
//    LocalResource { local_path: PathBuf::from("/models/Fake-News-Bert-Detect/merges.txt") };
//    true,
//    None,
//    None,
// );

// let sequence_classification_model = SequenceClassificationModel::new(config)?;
// let input = [
//     "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
//     "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
//     "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
// ];
// let output = sequence_classification_model.predict(&input);

#[tokio::main]
async fn main() {
    let sample_response = r#"
        {
            "cardiffnlp": {
                "score": 0.2,
                "label": "POSITIVE"
            },
            "ivanlau": {
                "score": 0.2,
                "label": "English"
            },
            "svalabs": {
                "score": 0.2,
                "label": "SPAM"
            },
            "EIStakovskii": {
                "score": 0.2,
                "label": "LABEL_0"
            },
            "jy46604790": {
                "score": 0.2,
                "label": "LABEL_0"
            }
        }"#;
    let sample_response_de: HashMap<String, ModelResult> = from_str(sample_response).unwrap(); 

    let process_handle = warp::post()
        .and(warp::path("process"))
        .and(warp::body::bytes())
        .map(move |_text_bytes: Bytes| {
            println!("bytes = {:?}", _text_bytes);
            warp::reply::json(&sample_response_de)
        });

    warp::serve(process_handle)
        .run(([0, 0, 0, 0], 3030))
        .await;
}