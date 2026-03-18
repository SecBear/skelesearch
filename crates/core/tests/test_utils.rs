/// Shared test utilities for skelesearch-core integration tests.
use async_trait::async_trait;

use skelesearch_core::EmbedProvider;

// ---------------------------------------------------------------------------
// DeterministicTestProvider
// ---------------------------------------------------------------------------

/// Returns normalised fixed vectors for every input without calling a model.
///
/// Vectors vary slightly per-position so FTS and vector scores can diverge in
/// tests, but the output is fully deterministic — identical inputs always
/// produce identical outputs.
#[derive(Clone)]
pub struct DeterministicTestProvider {
    dim: usize,
}

impl DeterministicTestProvider {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[async_trait]
impl EmbedProvider for DeterministicTestProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                // Slightly vary vectors per-chunk so FTS and vector scores differ.
                let mut v = vec![0.1_f32; self.dim];
                if !v.is_empty() {
                    v[0] = (i as f32 + 1.0) * 0.1;
                }
                // Normalise to unit length so cosine similarity is well-defined.
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// copy_dir_all
// ---------------------------------------------------------------------------

/// Recursively copy a directory tree from `src` to `dst`.
///
/// Used in integration tests to create isolated, writable snapshots of
/// on-disk fixture repos so tests cannot interfere with one another.
pub fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            std::fs::create_dir_all(&target)?;
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}
