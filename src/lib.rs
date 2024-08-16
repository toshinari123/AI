use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use warp::Filter;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use anyhow::Result;
use std::collections::HashMap;
use rand::Rng;
use reqwest;
use scraper::{Html, Selector};

pub struct LLM {
    name: String,
    markov_model: MarkovModel,
    bert_model: TextGenerationModel,
}

struct MarkovModel {
    transitions: HashMap<String, HashMap<char, usize>>,
    order: usize,
}

impl MarkovModel {
    fn new(order: usize) -> Self {
        MarkovModel {
            transitions: HashMap::new(),
            order,
        }
    }

    fn train(&mut self, text: &str) {
        for window in text.chars().collect::<Vec<char>>().windows(self.order + 1) {
            let state = window[..self.order].iter().collect::<String>();
            let next_char = window[self.order];
            
            self.transitions
                .entry(state)
                .or_insert_with(HashMap::new)
                .entry(next_char)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    fn generate(&self, seed: &str, length: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut result = seed.to_string();
        let mut current_state = seed.chars().rev().take(self.order).collect::<String>().chars().rev().collect::<String>();

        for _ in 0..length {
            if let Some(char_counts) = self.transitions.get(&current_state) {
                let total: usize = char_counts.values().sum();
                let mut random = rng.gen_range(0..total);
                
                for (char, &count) in char_counts {
                    if random < count {
                        result.push(*char);
                        current_state = format!("{}{}", &current_state[1..], char);
                        break;
                    }
                    random -= count;
                }
            } else {
                break;
            }
        }

        result
    }
}

impl LLM {
    pub fn new(name: &str) -> Result<Self> {
        Ok(LLM {
            name: name.to_string(),
            markov_model: MarkovModel::new(3),
            bert_model: TextGenerationModel::new(Default::default())?,
        })
    }

    pub fn train(&mut self, text: &str) {
        self.markov_model.train(text);
    }

    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let markov_result = self.markov_model.generate(prompt, 50);
        let bert_result = self.bert_model.generate(&[prompt], None)?;
        
        Ok(format!("{} says:\nMarkov: {}\nBERT: {}", 
                   self.name, 
                   markov_result, 
                   bert_result.join(" ")))
    }

    pub async fn train_from_wikipedia(&mut self, topic: &str) -> Result<()> {
        let url = format!("https://en.wikipedia.org/wiki/{}", topic);
        let resp = reqwest::get(&url).await?;
        let body = resp.text().await?;
        
        let document = Html::parse_document(&body);
        let selector = Selector::parse("p").unwrap();
        
        let text: String = document.select(&selector)
            .map(|element| element.text().collect::<String>())
            .collect::<Vec<String>>()
            .join(" ");

        self.train(&text);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct Query {
    prompt: String,
}

#[derive(Debug, Deserialize)]
pub struct TrainQuery {
    topic: String,
}

#[derive(Debug, Serialize)]
pub struct Response {
    result: String,
}

pub async fn run_server(llm: Arc<Mutex<LLM>>) {
    let llm = warp::any().map(move || llm.clone());

    let generate = warp::post()
        .and(warp::path("generate"))
        .and(warp::body::json())
        .and(llm.clone())
        .and_then(handle_generate);

    let train = warp::post()
        .and(warp::path("train"))
        .and(warp::body::json())
        .and(llm)
        .and_then(handle_train);

    let routes = generate.or(train);

    warp::serve(routes)
        .run(([127, 0, 0, 1], 3030))
        .await;
}

async fn handle_generate(
    query: Query,
    llm: Arc<Mutex<LLM>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let result = llm.lock().await.generate(&query.prompt).await.unwrap_or_else(|e| e.to_string());
    Ok(warp::reply::json(&Response { result }))
}

async fn handle_train(
    query: TrainQuery,
    llm: Arc<Mutex<LLM>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let result = llm.lock().await.train_from_wikipedia(&query.topic).await
        .map(|_| "Training completed successfully".to_string())
        .unwrap_or_else(|e| format!("Training failed: {}", e));
    Ok(warp::reply::json(&Response { result }))
}
