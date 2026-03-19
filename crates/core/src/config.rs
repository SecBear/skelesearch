use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub index: IndexConfig,
    pub search: SearchConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    pub provider: String,
    pub batch_size: usize,
    pub exclude: Vec<String>,
    pub index_dir: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub default_top_k: usize,
    /// Reranker configuration.
    #[serde(default)]
    pub reranker: RerankerConfig,
    /// Query expansion configuration.
    #[serde(default)]
    pub expansion: ExpansionConfig,
}

/// Reranker configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RerankerConfig {
    /// Reranker provider: "jina", "cohere", "voyage", or absent for auto-detect.
    #[serde(default)]
    pub provider: Option<String>,
    /// Environment variable name for the API key (default: auto-detect per provider).
    #[serde(default)]
    pub api_key_env: Option<String>,
}

/// Query expansion configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ExpansionConfig {
    /// Enable LLM query expansion.  Absent = auto-detect from OPENAI_API_KEY.
    #[serde(default)]
    pub enabled: Option<bool>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self { default_top_k: 5, reranker: RerankerConfig::default(), expansion: ExpansionConfig::default() }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self { provider: "fastembed".into(), batch_size: 64, exclude: vec![], index_dir: None }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self { index: IndexConfig::default(), search: SearchConfig::default() }
    }
}

impl Config {
    /// Parse a `Config` from a TOML string.  Returns an error on invalid TOML;
    /// unknown keys are ignored (serde `#[serde(default)]`).
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(s)?)
    }

    /// Load config from `<project_root>/.skelesearch.toml`, returning defaults
    /// when the file is absent.  Returns an error only on I/O or parse failure.
    pub fn load(project_root: &Path) -> anyhow::Result<Self> {
        let path = project_root.join(".skelesearch.toml");
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            Self::from_str(&content)
        } else {
            Ok(Self::default())
        }
    }
}
