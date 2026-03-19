//! Evaluation harness for measuring search quality.

use serde::Deserialize;

/// A single eval case: a query with known-relevant file paths.
#[derive(Debug, Deserialize)]
pub struct EvalCase {
    pub query: String,
    /// File paths (relative to project root) that are relevant to this query.
    pub expected_files: Vec<String>,
}

/// Metrics for a single eval case.
#[derive(Debug)]
pub struct CaseMetrics {
    pub query: String,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub retrieved_files: Vec<String>,
}

/// Aggregate metrics across all eval cases.
#[derive(Debug)]
pub struct AggregateMetrics {
    pub mean_recall_at_5: f64,
    pub mean_recall_at_10: f64,
    pub mean_mrr: f64,
    pub total_cases: usize,
}

/// Compute recall@K: fraction of expected files in the top K retrieved.
pub fn recall_at_k(retrieved: &[String], expected: &[String], k: usize) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let top_k: Vec<&str> = retrieved.iter().take(k).map(|s| s.as_str()).collect();
    let hits = expected
        .iter()
        .filter(|e| top_k.iter().any(|r| r.ends_with(e.as_str()) || e.ends_with(r)))
        .count();
    hits as f64 / expected.len() as f64
}

/// Compute MRR: reciprocal rank of the first relevant result.
pub fn mrr(retrieved: &[String], expected: &[String]) -> f64 {
    for (i, r) in retrieved.iter().enumerate() {
        if expected
            .iter()
            .any(|e| r.ends_with(e.as_str()) || e.ends_with(r.as_str()))
        {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

/// Compute aggregate metrics from per-case metrics.
pub fn aggregate(cases: &[CaseMetrics]) -> AggregateMetrics {
    let n = cases.len().max(1) as f64;
    AggregateMetrics {
        mean_recall_at_5: cases.iter().map(|c| c.recall_at_5).sum::<f64>() / n,
        mean_recall_at_10: cases.iter().map(|c| c.recall_at_10).sum::<f64>() / n,
        mean_mrr: cases.iter().map(|c| c.mrr).sum::<f64>() / n,
        total_cases: cases.len(),
    }
}
