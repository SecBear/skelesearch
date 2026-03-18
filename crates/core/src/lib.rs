pub mod chunker;
pub mod indexer;
pub mod manifest;
pub mod provider;
pub mod schema;
pub mod searcher;

// Re-export the most-used public types so callers can write
// `use skelesearch_core::CozoBackend` instead of the full path.
pub use chunker::{normalize_for_fts, Chunker, ImportEdge, ParsedChunk};
pub use indexer::{IndexResult, Indexer};
pub use manifest::ManifestStore;
pub use provider::EmbedProvider;
pub use schema::{
    ChunkRecord, CozoBackend, EdgeRecord, FileRecord, IndexStats, SearchResult, StorageBackend,
};
pub use searcher::{FileContext, Searcher};