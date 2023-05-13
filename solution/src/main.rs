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
use tokio::{sync::oneshot, task};
use serde::{Deserialize, Serialize};
use warp::Filter;

const MODEL_DIR_NAME_FAKE_NEWS: &'static str = "Fake-News-Bert-Detect";
const MODEL_DIR_NAME_LANGUAGE: &'static str = "language-detection-fine-tuned-on-xlm-roberta-base";
const MODEL_DIR_NAME_SENTIMENT: &'static str = "twitter-xlm-roberta-base-sentiment";
const MODEL_DIR_NAME_SPAM: &'static str = "twitter-xlm-roberta-crypto-spam";
const MODEL_DIR_NAME_TOXICITY: &'static str = "xlm_roberta_base_multilingual_toxicity_classifier_plus";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelResult {
    pub score: f64,
    pub label: String,
}

type Message = (Vec<String>, oneshot::Sender<Vec<Label>>);

#[derive(Debug)]
pub struct SequenceClassifierProcessor {
    handle: JoinHandle<Result<()>>,
    sender: mpsc::SyncSender<Message>,
}

impl SequenceClassifierProcessor {
    fn new(model_dir_name: String) -> SequenceClassifierProcessor {
        let (sender, receiver) = mpsc::sync_channel(100);
        let handle = thread::spawn(move || Self::runner(model_dir_name, receiver));
        SequenceClassifierProcessor {
            handle: handle,
            sender: sender
        }
    }

    fn runner(model_dir_name: String, receiver: mpsc::Receiver<Message>) -> Result<()> {
        let config: SequenceClassificationConfig;
        if model_dir_name == MODEL_DIR_NAME_FAKE_NEWS {
            let model_resource = LocalResource {
                local_path: PathBuf::from(format!("/models/{}/rust_model.ot", model_dir_name))
            };
            let config_resource = LocalResource {
                local_path: PathBuf::from(format!("/models/{}/config.json", model_dir_name))
            };
            let vocab_resource = LocalResource {
                local_path: PathBuf::from(format!("/models/{}/vocab.json", model_dir_name))
            };
            let merges_resource = LocalResource {
                local_path: PathBuf::from(format!("/models/{}/merges.txt", model_dir_name))
            };

            config = SequenceClassificationConfig::new(
               ModelType::Roberta,
               model_resource,
               config_resource,
               vocab_resource,
               Some(merges_resource),
               true,
               None,
               None
            );
        } else {
            let model_resource = LocalResource {
                local_path: PathBuf::from(format!("/models/{}/rust_model.ot", model_dir_name))
            };
            let config_resource = LocalResource {
                local_path: PathBuf::from(format!("/models/{}/config.json", model_dir_name))
            };
            let vocab_resource = LocalResource {
                local_path: PathBuf::from(format!("/models/{}/sentencepiece.bpe.model", model_dir_name))
            };

            config = SequenceClassificationConfig::new(
               ModelType::XLMRoberta,
               model_resource,
               config_resource,
               vocab_resource,
               None,
               true,
               None,
               None
            );
        }

        let model = SequenceClassificationModel::new(config).unwrap();

        while let Ok((texts, sender)) = receiver.recv() {
            let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
            let sentiments = model.predict(texts);
            sender.send(sentiments).expect("sending results");
        }
        
        Ok(())
    }

    // async fn predict(&self, texts: Vec<String>) -> Result<Vec<Label>> {
    //     let (sender, receiver) = oneshot::channel();
    //     task::block_in_place(|| self.sender.send((texts, sender)))?;
    //     Ok(receiver.await?)
    // }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model_dir_names = HashMap::from([
        (String::from("jy46604790"), MODEL_DIR_NAME_FAKE_NEWS.to_owned()),
        (String::from("ivanlau"), MODEL_DIR_NAME_LANGUAGE.to_owned()),
        (String::from("cardiffnlp"), MODEL_DIR_NAME_SENTIMENT.to_owned()),
        (String::from("svalabs"), MODEL_DIR_NAME_SPAM.to_owned()),
        (String::from("EIStakovskii"), MODEL_DIR_NAME_TOXICITY.to_owned()),
    ]);

    let mut processors: Vec<SequenceClassifierProcessor> = Vec::new();
    let mut processors_channels: HashMap<String, mpsc::SyncSender<Message>> = HashMap::new();
    for (name, dir) in &model_dir_names {
        let p = SequenceClassifierProcessor::new(dir.to_string());
        processors_channels.insert(name.clone(), p.sender.clone());
        processors.push(p);
    }
    
    let with_channels = warp::any().map(move || processors_channels.clone());
    let process_handle = warp::post()
        .and(warp::path("process"))
        .and(warp::body::bytes())
        .and(with_channels)
        .and_then(|text: Bytes, pc: HashMap<String, mpsc::SyncSender<Message>>| async move {
            println!("bytes = {:?}", text);
            let mut final_result: HashMap<String, ModelResult> = HashMap::new(); 
            let texts = vec![String::from_utf8(text.to_vec()).unwrap()];
            for (name, c) in &pc {
                let (sender, receiver) = oneshot::channel();
                task::block_in_place(|| c.send((texts.clone(), sender))).unwrap();
                let res = receiver.await.unwrap();
                final_result.insert(name.to_string(), ModelResult {score: res[0].score.clone(), label: res[0].text.clone()});
            }
            println!("Results: {final_result:?}");
            // Ok::<_, warp::Rejection>(format!("Hello #{}", "1"))
            Ok::<_, warp::Rejection>(warp::reply::json(&final_result))
        });

    warp::serve(process_handle)
        .run(([0, 0, 0, 0], 3030))
        .await;
    
    Ok(())
}
