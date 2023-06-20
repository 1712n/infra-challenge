use std::collections::{HashMap};
use std::path::PathBuf;
use std::thread::{self, JoinHandle};

use anyhow::Result;
use bytes::Bytes;
use rust_bert::pipelines::sequence_classification::SequenceClassificationConfig;
use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
use rust_bert::pipelines::sequence_classification::Label;
use rust_bert::pipelines::common::ModelType;
use rust_bert::resources::LocalResource;
use tokio::{sync::oneshot, sync::mpsc};
use tokio::time::{sleep, Duration};
use serde::{Deserialize, Serialize};
use warp::Filter;

const MODEL_USER_NAME_FAKE_NEWS: &'static str = "jy46604790";
const MODEL_USER_NAME_LANGUAGE: &'static str = "ivanlau";
const MODEL_USER_NAME_SENTIMENT: &'static str = "cardiffnlp";
const MODEL_USER_NAME_SPAM: &'static str = "svalabs";
const MODEL_USER_NAME_TOXICITY: &'static str = "EIStakovskii";

const MODEL_DIR_NAME_FAKE_NEWS: &'static str = "Fake-News-Bert-Detect";
const MODEL_DIR_NAME_LANGUAGE: &'static str = "language-detection-fine-tuned-on-xlm-roberta-base";
const MODEL_DIR_NAME_SENTIMENT: &'static str = "twitter-xlm-roberta-base-sentiment";
const MODEL_DIR_NAME_SPAM: &'static str = "twitter-xlm-roberta-crypto-spam";
const MODEL_DIR_NAME_TOXICITY: &'static str = "xlm_roberta_base_multilingual_toxicity_classifier_plus";

const DISPATCH_CHANNEL_CAPACITY: usize = 70;
const DISPATCH_CHANNEL_TRY_RECV_WAIT_MILLIS: u64 = 40;
const INFERENCE_BATCH_SIZE: usize = 3;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelResult {
    pub score: f64,
    pub label: String,
}

type ModelInferenceInput = (Vec<String>, oneshot::Sender<Vec<Label>>);
type ModelRunnerSenderMap = HashMap<String, mpsc::UnboundedSender<ModelInferenceInput>>;
type Response = HashMap<String, ModelResult>;
type DispatchMessage = (String, oneshot::Sender<Response>);

fn run_model(model_dir_name: String, mut receiver: mpsc::UnboundedReceiver<ModelInferenceInput>) -> Result<()> {
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

    while let Some((texts, sender)) = receiver.blocking_recv() {
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        let result = model.predict(texts);
        sender.send(result).expect("sending results");
    }
    
    Ok(())
}

fn spawn_model_runner_thread(model_dir_name: String) -> (JoinHandle<Result<()>>, mpsc::UnboundedSender<ModelInferenceInput>) {
    let (sender, receiver): (mpsc::UnboundedSender<ModelInferenceInput>, mpsc::UnboundedReceiver<ModelInferenceInput>) = mpsc::unbounded_channel();
    let handle = thread::spawn(move || run_model(model_dir_name, receiver));
    (handle, sender)
}

fn spawn_dispatch_task(model_runner_sender_map: ModelRunnerSenderMap) -> (mpsc::Sender<DispatchMessage>, tokio::task::JoinHandle<()>) {
    let (dispatch_sender, mut dispatch_receiver) = mpsc::channel(DISPATCH_CHANNEL_CAPACITY);
    let dispatch_sender_clone = dispatch_sender.clone();
    let dispatch_task_handle = tokio::spawn(async move {
        let mut input_texts: Vec<String> = Vec::with_capacity(INFERENCE_BATCH_SIZE);
        let mut input_senders: Vec<oneshot::Sender<HashMap<String, ModelResult>>> = Vec::with_capacity(INFERENCE_BATCH_SIZE);
        loop {
            let mut recv_tries_count = 0;
            loop {
                match dispatch_receiver.try_recv() {
                    Ok((input_text, input_sender)) => {
                        input_texts.push(input_text);
                        input_senders.push(input_sender);
                    },
                    Err(_) => {
                        break;
                    }
                }
                recv_tries_count += 1;
                if recv_tries_count == INFERENCE_BATCH_SIZE {
                    break;
                }
            }
            if recv_tries_count == 0 {
                sleep(Duration::from_millis(DISPATCH_CHANNEL_TRY_RECV_WAIT_MILLIS)).await;
                continue;
            }
            let mut output_responses: Vec<HashMap<String, ModelResult>> = Vec::with_capacity(INFERENCE_BATCH_SIZE); 
            for _ in 0..INFERENCE_BATCH_SIZE {
                let response: HashMap<String, ModelResult> = HashMap::with_capacity(5); 
                output_responses.push(response);
            }
            let mut model_receivers = Vec::with_capacity(5);
            for (name, model_runner_sender) in &model_runner_sender_map {
                let (model_sender, model_receiver) = oneshot::channel();
                model_runner_sender.send((input_texts.clone(), model_sender)).unwrap();
                model_receivers.push((name.to_string(), model_receiver));
            }
            for (name, receiver) in model_receivers {
                let model_output = receiver.await.unwrap();
                for (i, out) in model_output.into_iter().enumerate() {
                    if name == "cardiffnlp" {
                        output_responses[i].insert(name.to_string(), ModelResult {score: out.score.clone(), label: out.text.clone().to_uppercase()});
                    } else {
                        output_responses[i].insert(name.to_string(), ModelResult {score: out.score.clone(), label: out.text.clone()});
                    }
                }
            }
            for i in 0..INFERENCE_BATCH_SIZE {
                if input_senders.len() >= 1 {
                    input_senders.remove(0).send(output_responses[i].clone()).unwrap();
                } else {
                    break;
                }
            }
            input_texts.clear();
            input_senders.clear();
        }
    });
    (dispatch_sender_clone, dispatch_task_handle)
}

#[tokio::main]
async fn main() -> Result<()> {
    let model_dir_names = HashMap::from([
        (MODEL_USER_NAME_FAKE_NEWS.to_owned(), MODEL_DIR_NAME_FAKE_NEWS.to_owned()),
        (MODEL_USER_NAME_LANGUAGE.to_owned(), MODEL_DIR_NAME_LANGUAGE.to_owned()),
        (MODEL_USER_NAME_SENTIMENT.to_owned(), MODEL_DIR_NAME_SENTIMENT.to_owned()),
        (MODEL_USER_NAME_SPAM.to_owned(), MODEL_DIR_NAME_SPAM.to_owned()),
        (MODEL_USER_NAME_TOXICITY.to_owned(), MODEL_DIR_NAME_TOXICITY.to_owned()),
    ]);

    let mut _handles: Vec<JoinHandle<Result<()>>> = Vec::with_capacity(5);
    let mut model_runner_sender_map: HashMap<String, mpsc::UnboundedSender<ModelInferenceInput>> = HashMap::with_capacity(5);
    for (name, dir) in &model_dir_names {
        let (handle, model_mpsc_sender) = spawn_model_runner_thread(dir.to_string());
        model_runner_sender_map.insert(name.clone(), model_mpsc_sender.clone());
        _handles.push(handle);
    }

    let (dispatch_sender, _dispatch_task_handle) = spawn_dispatch_task(model_runner_sender_map);

    let with_dispatch_sender = warp::any().map(move || {
        if dispatch_sender.capacity() == 0 {
            return Err::<_, warp::Rejection>(warp::reject::reject())
        }
        Ok(dispatch_sender.clone())
    });
    let process_handle = warp::post()
        .and(warp::path("process"))
        .and(with_dispatch_sender)
        .and(warp::body::bytes())
        .and_then(|dispatch_sender: Result<mpsc::Sender<DispatchMessage>, warp::Rejection>, text_input: Bytes| async move {
            match dispatch_sender {
                Ok(v) => {
                    let input_text = String::from_utf8(text_input.to_vec()).unwrap();
                    let (sender, receiver) = oneshot::channel();
                    v.send((input_text, sender)).await.unwrap();
                    let response = receiver.await.unwrap();
                    Ok::<_, warp::Rejection>(warp::reply::json(&response))
                },
                Err(_) => return Ok::<_, warp::Rejection>(warp::reply::json(&String::from("{}")))

            }
        });

    warp::serve(process_handle)
        .run(([0, 0, 0, 0], 3030))
        .await;
    
    Ok(())
}
