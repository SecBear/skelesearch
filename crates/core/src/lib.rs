pub mod sparse;
pub use sparse::{SparseEmbedding, SparseEmbedProvider};

pub mod cochange;
pub use cochange::CoChangePair;

pub mod summary;
pub use summary::{NoopSummaryProvider, OpenAISummaryProvider, SummaryProvider};

pub mod git;

pub mod chunker;
pub mod indexer;
pub mod manifest;
pub mod provider;
pub mod schema;
pub mod searcher;
pub mod gc;
pub mod grep;
pub mod config;

// Re-export the most-used public types so callers can write
// `use skelesearch_core::CozoBackend` instead of the full path.
pub use chunker::{normalize_for_fts, Chunker, ImportEdge, ParsedChunk};
pub use indexer::{FileContent, IndexResult, Indexer};
pub use manifest::{content_hash, IncompleteBatch, ManifestStore};
pub use provider::EmbedProvider;
pub use schema::{
    CallEdge, ChunkRecord, CozoBackend, EdgeRecord, FileRecord, IndexStats, RepoMapData, RepoMapFile,
    RepoMapSymbol, SearchResult, StorageBackend,
};
pub use searcher::{FileContext, Searcher, SearchTimings};
pub use gc::collect_garbage;
pub use grep::{grep_codebase, GrepMatch, GrepOptions};
pub use config::{Config, ExpansionConfig, GraphConfig, IndexConfig, RerankerConfig, SearchConfig, SparseConfig};
pub mod reranker;
pub use reranker::{NoopReranker, RerankCandidate, Reranker};
pub mod router;
pub use router::{classify_query, QueryStrategy};
pub mod symbols;
pub use symbols::{extract_references, extract_symbols, ReferenceCapture, SymbolDef};
pub mod expander;
pub use expander::{LLMExpander, NoopExpander, QueryExpander};

pub mod resolve;
pub use resolve::{extract_import_path, resolver_for_extension, ImportResolver};