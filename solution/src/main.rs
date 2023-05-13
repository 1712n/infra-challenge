use std::collections::{HashMap};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use anyhow::Result;
use bytes::Bytes;
use rust_bert::pipelines::sequence_classification::SequenceClassificationConfig;
use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
use rust_bert::pipelines::sequence_classification::Label;
use rust_bert::pipelines::common::ModelType;
use rust_bert::resources::LocalResource;
use tch::Device;
use tokio::{sync::oneshot, task};
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
// 


// use rust_bert::pipelines::sentiment::{Sentiment, SentimentConfig, SentimentModel};

// #[tokio::main]
// async fn main() -> Result<()> {
//     let (_handle, classifier) = SentimentClassifier::spawn();

//     let texts = vec![
//         "Classify this positive text".to_owned(),
//         "Classify this negative text".to_owned(),
//     ];
//     let sentiments = classifier.predict(texts).await?;
//     println!("Results: {sentiments:?}");

//     Ok(())
// }

/// Message type for internal channel, passing around texts and return value
/// senders
type Message = (Vec<String>, oneshot::Sender<Vec<Label>>);

/// Runner for sentiment classification
#[derive(Debug, Clone)]
pub struct SentimentClassifier {
    sender: mpsc::SyncSender<Message>,
}

impl SentimentClassifier {
    /// Spawn a classifier on a separate thread and return a classifier instance
    /// to interact with it
    pub fn spawn() -> (JoinHandle<Result<()>>, SentimentClassifier) {
        let (sender, receiver) = mpsc::sync_channel(100);
        println!("6");
        let handle = thread::spawn(move || Self::runner(receiver));
        (handle, SentimentClassifier { sender })
    }

    /// The classification runner itself
    fn runner(receiver: mpsc::Receiver<Message>) -> Result<()> {
        // Needs to be in sync runtime, async doesn't work
        // let model = SentimentModel::new(SentimentConfig::default())?;
        // println!("4");
        // let model_resource = LocalResource {
        //     local_path: PathBuf::from("/models/Fake-News-Bert-Detect/rust_model.ot"),
        // };
        // let vocab_resource = LocalResource {
        //     local_path: PathBuf::from("/models/Fake-News-Bert-Detect/vocab.json"),
        // };
        // let config_resource = LocalResource {
        //     local_path: PathBuf::from("/models/Fake-News-Bert-Detect/config.json"),
        // };
        // let merges_resource = LocalResource {
        //     local_path: PathBuf::from("/models/Fake-News-Bert-Detect/merges.txt"),
        // };
        // println!("7");

        // let config = SequenceClassificationConfig::new(
        //    ModelType::Roberta,
        //    model_resource,
        //    config_resource,
        //    vocab_resource,
        //    Some(merges_resource),
        //    true,
        //    None,
        //    None
        // );

        // println!("4");
        // let model_resource = LocalResource {
        //     local_path: PathBuf::from("/models/xlm_roberta_base_multilingual_toxicity_classifier_plus/rust_model.ot"),
        // };
        // let vocab_resource = LocalResource {
        //     local_path: PathBuf::from("/models/xlm_roberta_base_multilingual_toxicity_classifier_plus/sentencepiece.bpe.model"),
        // };
        // let config_resource = LocalResource {
        //     local_path: PathBuf::from("/models/xlm_roberta_base_multilingual_toxicity_classifier_plus/config.json"),
        // };
        // println!("7");

        // let config = SequenceClassificationConfig::new(
        //    ModelType::XLMRoberta,
        //    model_resource,
        //    config_resource,
        //    vocab_resource,
        //    None,
        //    true,
        //    None,
        //    None
        // );
        // println!("7.5");

        // println!("4");
        // let model_resource = LocalResource {
        //     local_path: PathBuf::from("/models/twitter-xlm-roberta-crypto-spam/rust_model.ot"),
        // };
        // let vocab_resource = LocalResource {
        //     local_path: PathBuf::from("/models/twitter-xlm-roberta-crypto-spam/sentencepiece.bpe.model"),
        // };
        // let config_resource = LocalResource {
        //     local_path: PathBuf::from("/models/twitter-xlm-roberta-crypto-spam/config.json"),
        // };
        // println!("7");

        // let config = SequenceClassificationConfig::new(
        //    ModelType::XLMRoberta,
        //    model_resource,
        //    config_resource,
        //    vocab_resource,
        //    None,
        //    true,
        //    None,
        //    None
        // );
        // println!("7.5");

        // let model = SequenceClassificationModel::new(config).unwrap();
        // println!("8");

        // println!("4");
        // let model_resource = LocalResource {
        //     local_path: PathBuf::from("/models/language-detection-fine-tuned-on-xlm-roberta-base/rust_model.ot"),
        // };
        // let vocab_resource = LocalResource {
        //     local_path: PathBuf::from("/models/language-detection-fine-tuned-on-xlm-roberta-base/sentencepiece.bpe.model"),
        // };
        // let config_resource = LocalResource {
        //     local_path: PathBuf::from("/models/language-detection-fine-tuned-on-xlm-roberta-base/config.json"),
        // };
        // println!("7");

        // let config = SequenceClassificationConfig::new(
        //    ModelType::XLMRoberta,
        //    model_resource,
        //    config_resource,
        //    vocab_resource,
        //    None,
        //    true,
        //    None,
        //    None
        // );
        // println!("7.5");

        println!("4");
        let model_resource = LocalResource {
            local_path: PathBuf::from("/models/twitter-xlm-roberta-base-sentiment/rust_model.ot"),
        };
        let vocab_resource = LocalResource {
            local_path: PathBuf::from("/models/twitter-xlm-roberta-base-sentiment/sentencepiece.bpe.model"),
        };
        let config_resource = LocalResource {
            local_path: PathBuf::from("/models/twitter-xlm-roberta-base-sentiment/config.json"),
        };
        println!("7");

        let config = SequenceClassificationConfig::new(
           ModelType::XLMRoberta,
           model_resource,
           config_resource,
           vocab_resource,
           None,
           true,
           None,
           None
        );
        println!("7.5");


        let model = SequenceClassificationModel::new(config).unwrap();
        println!("8");

        while let Ok((texts, sender)) = receiver.recv() {
            println!("9");
            let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
            println!("10");
            let sentiments = model.predict(texts);
            println!("11");
            sender.send(sentiments).expect("sending results");
        }
        Ok(())
    }

    /// Make the runner predict a sample and return the result
    pub async fn predict(&self, texts: Vec<String>) -> Result<Vec<Label>> {
        let (sender, receiver) = oneshot::channel();
        println!("5");
        task::block_in_place(|| self.sender.send((texts, sender)))?;
        Ok(receiver.await?)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("1");
    let (_handle, classifier) = SentimentClassifier::spawn();

    println!("2");
    let texts = vec![
        "Classify this positive text".to_owned(),
        "Classify this negative text".to_owned(),
    ];

    println!("3");
    let sentiments = classifier.predict(texts).await?;
    println!("Results: {sentiments:?}");

    // let sample_response = r#"
    //     {
    //         "cardiffnlp": {
    //             "score": 0.2,
    //             "label": "POSITIVE"
    //         },
    //         "ivanlau": {
    //             "score": 0.2,
    //             "label": "English"
    //         },
    //         "svalabs": {
    //             "score": 0.2,
    //             "label": "SPAM"
    //         },
    //         "EIStakovskii": {
    //             "score": 0.2,
    //             "label": "LABEL_0"
    //         },
    //         "jy46604790": {
    //             "score": 0.2,
    //             "label": "LABEL_0"
    //         }
    //     }"#;
    // let sample_response_de: HashMap<String, ModelResult> = from_str(sample_response).unwrap(); 

    // let process_handle = warp::post()
    //     .and(warp::path("process"))
    //     .and(warp::body::bytes())
    //     .map(move |_text_bytes: Bytes| {
    //         println!("bytes = {:?}", _text_bytes);
    //         warp::reply::json(&sample_response_de)
    //     });

    // warp::serve(process_handle)
    //     .run(([0, 0, 0, 0], 3030))
    //     .await;
    
    Ok(())
}