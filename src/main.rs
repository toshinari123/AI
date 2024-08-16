use llm_host::{LLM, run_server};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::new("MyLLM")?;
    let llm = Arc::new(Mutex::new(llm));
    run_server(llm).await;

    Ok(())
}
