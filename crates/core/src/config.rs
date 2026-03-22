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
    /// Append extracted symbol names to each chunk's normalised text for BM25.
    /// Disable to measure the contribution of symbol enrichment in benchmarks.
    pub symbol_enrichment: bool,
    /// Override the default file-extension allowlist.  When set, only files
    /// whose extension appears in this list are indexed.  When absent, the
    /// built-in allowlist in `indexer.rs` is used.
    pub include_extensions: Option<Vec<String>>,
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
    /// Graph / import-edge traversal configuration.
    #[serde(default)]
    pub graph: GraphConfig,
}

/// Reranker configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RerankerConfig {
    /// Reranker provider: "jina", "cohere", "voyage", "local", or absent for auto-detect.
    #[serde(default)]
    pub provider: Option<String>,
    /// Environment variable name for the API key (default: auto-detect per provider).
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Path to local ONNX model directory (for provider = "local").
    /// If absent, uses default cache directory.
    #[serde(default)]
    pub model_dir: Option<String>,
}

/// Query expansion configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ExpansionConfig {
    /// Enable LLM query expansion.  Absent = auto-detect from OPENAI_API_KEY.
    #[serde(default)]
    pub enabled: Option<bool>,
}

/// Graph search (import-edge traversal) configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GraphConfig {
    /// Enable graph search.  Absent / `None` means disabled unless the CLI `--graph` flag
    /// is passed.  Set to `true` to enable for all callers including `eval`.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Maximum edge-traversal depth when graph search is active.
    pub max_depth: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self { enabled: None, max_depth: 1 }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_top_k: 5,
            reranker: RerankerConfig::default(),
            expansion: ExpansionConfig::default(),
            graph: GraphConfig::default(),
        }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            provider: "fastembed".into(),
            batch_size: 64,
            exclude: vec![],
            index_dir: None,
            symbol_enrichment: true,
            include_extensions: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self { index: IndexConfig::default(), search: SearchConfig::default() }
    }
}

impl Config {
    /// Parse a `Config` from a TOML string.  Returns an error on invalid TOML;
    /// unknown keys are ignored (serde `#[serde(default)]`).  `batch_size = 0`
    /// is clamped to 1 so callers never receive a zero-sized batch.
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        let mut cfg: Self = toml::from_str(s)?;
        cfg.index.batch_size = cfg.index.batch_size.max(1);
        Ok(cfg)
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_current_behavior() {
        let cfg = Config::default();
        assert!(cfg.index.symbol_enrichment, "symbol enrichment must be on by default");
        assert_eq!(cfg.search.graph.enabled, None, "graph must be disabled by default");
        assert_eq!(cfg.search.graph.max_depth, 1);
    }

    #[test]
    fn parse_graph_config() {
        let toml = r#"
[search.graph]
enabled = true
max_depth = 2
"#;
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.search.graph.enabled, Some(true));
        assert_eq!(cfg.search.graph.max_depth, 2);
        // Unspecified index fields should remain at defaults.
        assert!(cfg.index.symbol_enrichment);
    }

    #[test]
    fn parse_symbol_enrichment_false() {
        let toml = r#"
[index]
symbol_enrichment = false
"#;
        let cfg = Config::from_str(toml).unwrap();
        assert!(!cfg.index.symbol_enrichment);
        // Graph should remain at default.
        assert_eq!(cfg.search.graph.enabled, None);
    }

    #[test]
    fn parse_no_graph_toml() {
        // Simulates loading benchmarks/configs/no-graph.toml
        let toml = r#"
[index]
symbol_enrichment = true

[search.graph]
enabled = false
"#;
        let cfg = Config::from_str(toml).unwrap();
        assert!(cfg.index.symbol_enrichment);
        assert_eq!(cfg.search.graph.enabled, Some(false));
    }
}