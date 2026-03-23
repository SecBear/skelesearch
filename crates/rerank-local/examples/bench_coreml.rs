
use skelesearch_rerank_local::LocalReranker;
use skelesearch_core::reranker::{Reranker, RerankCandidate};
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // tracing setup would go here

    let model_dir = std::env::var("MODEL_DIR")
        .unwrap_or_else(|_| {
            let home = dirs::home_dir().unwrap();
            home.join(".cache/skelesearch/reranker/gte-modernbert-base")
                .to_string_lossy().to_string()
        });

    println!("Loading model from: {model_dir}");
    let t0 = Instant::now();
    let reranker = LocalReranker::new(&model_dir)?
        .with_max_seq_len(1024);
    println!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let query = "How does the server handle graceful shutdown";
    let candidates: Vec<RerankCandidate> = (0..10).map(|i| RerankCandidate {
        index: i,
        text: format!("pub async fn shutdown(signal: impl Future) -> Result<()> {{ tokio::select! {{ _ = signal => {{ info!('shutdown signal received'); }} }} }} // chunk {i}"),
    }).collect();

    // Warmup
    println!("Warmup...");
    let _ = reranker.rerank(query, candidates.clone()).await?;

    // Timed runs
    let n = 5;
    let mut times = Vec::new();
    for i in 0..n {
        let t = Instant::now();
        let scores = reranker.rerank(query, candidates.clone()).await?;
        let elapsed = t.elapsed().as_millis();
        times.push(elapsed);
        println!("Run {}: {}ms (scores[0]={:.4})", i + 1, elapsed, scores[0]);
    }

    let avg = times.iter().sum::<u128>() / n;
    let min = times.iter().min().unwrap();
    println!("\nAvg: {avg}ms  Min: {min}ms  (10 candidates, 1024 max_seq_len)");
    Ok(())
}
