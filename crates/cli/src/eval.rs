//! Evaluation harness for measuring search quality.

use serde::Deserialize;

/// A single eval case: a query with known-relevant file paths.
#[derive(Debug, Deserialize)]
pub struct EvalCase {
    pub query: String,
    /// File paths (relative to project root) that are relevant to this query.
    pub expected_files: Vec<String>,
    /// Optional query category for stratified analysis.
    #[serde(default)]
    pub category: Option<String>,
}

/// Metrics for a single eval case.
#[derive(Debug)]
pub struct CaseMetrics {
    pub query: String,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub precision_at_5: f64,
    pub mrr: f64,
    pub retrieved_files: Vec<String>,
    pub category: Option<String>,
}

/// Aggregate metrics across all eval cases.
#[derive(Debug)]
pub struct AggregateMetrics {
    pub mean_recall_at_5: f64,
    pub mean_recall_at_10: f64,
    pub mean_precision_at_5: f64,
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

/// Compute precision@K: fraction of top K results that are relevant.
pub fn precision_at_k(retrieved: &[String], expected: &[String], k: usize) -> f64 {
    let top_k: Vec<&str> = retrieved.iter().take(k).map(|s| s.as_str()).collect();
    if top_k.is_empty() {
        return 0.0;
    }
    let hits = top_k
        .iter()
        .filter(|r| expected.iter().any(|e| r.ends_with(e.as_str()) || e.ends_with(*r)))
        .count();
    hits as f64 / top_k.len() as f64
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
        mean_precision_at_5: cases.iter().map(|c| c.precision_at_5).sum::<f64>() / n,
        mean_mrr: cases.iter().map(|c| c.mrr).sum::<f64>() / n,
        total_cases: cases.len(),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_at_k_basic() {
        let retrieved = vec!["a.rs".into(), "b.rs".into(), "c.rs".into(), "d.rs".into(), "e.rs".into()];
        let expected = vec!["a.rs".into(), "c.rs".into()];
        // 2 of 5 top results are relevant
        assert!((precision_at_k(&retrieved, &expected, 5) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn precision_at_k_empty_retrieved() {
        let retrieved: Vec<String> = vec![];
        let expected = vec!["a.rs".into()];
        assert!((precision_at_k(&retrieved, &expected, 5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn precision_at_k_all_relevant() {
        let retrieved = vec!["a.rs".into(), "b.rs".into()];
        let expected = vec!["a.rs".into(), "b.rs".into()];
        assert!((precision_at_k(&retrieved, &expected, 5) - 1.0).abs() < 1e-9);
    }
}