pub mod sparse;
pub use sparse::{SparseEmbedProvider, SparseEmbedding};

pub mod cochange;
pub use cochange::CoChangePair;

pub mod summary;
pub use summary::{NoopSummaryProvider, OpenAISummaryProvider, SummaryProvider};

pub mod git;

pub mod chunker;
pub mod config;
pub mod coordination;
pub mod freshness;
pub mod gc;
pub mod grep;
pub mod indexer;
pub mod manifest;
pub mod provider;
pub mod schema;
pub mod searcher;

// Re-export the most-used public types so callers can write
// `use skelesearch_core::CompositeBackend` instead of the full path.
pub use chunker::{normalize_for_fts, Chunker, ImportEdge, ParsedChunk};
pub use config::{
    Config, ExpansionConfig, GraphConfig, IndexConfig, RerankerConfig, SearchConfig, SparseConfig,
};
pub use coordination::{
    is_indexing_active_elsewhere, read_shared_indexing_status, try_acquire_indexing_lease,
    write_file_atomic, IndexingLease, SharedIndexingStatus,
};
pub use freshness::{FreshnessSnapshot, FreshnessState};
pub use gc::collect_garbage;
pub use grep::{grep_codebase, GrepMatch, GrepOptions};
pub use indexer::{FileContent, IndexResult, Indexer};
pub use manifest::{content_hash, IncompleteBatch, ManifestStore};
pub use provider::{preferred_index_provider_name, EmbedProvider};
pub use schema::{
    generation_db_paths, CallEdge, ChunkRecord, EdgeRecord, FileRecord, IndexStats, RepoMapData,
    RepoMapFile, RepoMapSymbol, SearchResult, StorageBackend, INDEX_DB_FILE, MANIFEST_DB_FILE,
};
pub use searcher::{FileContext, SearchTimings, Searcher};
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

pub mod backend;
pub use backend::CompositeBackend;
