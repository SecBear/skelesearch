/// Arrow schema definitions for all LanceDB tables.
///
/// All schemas must use exactly the same arrow-array version as lancedb (57.x).
/// Embedding dimensions are runtime-configurable via `dim` — never hardcoded.
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

pub fn files_schema() -> Schema {
    Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("last_modified", DataType::Int64, false),
        Field::new("last_indexed", DataType::Int64, false),
        Field::new("chunk_count", DataType::UInt64, false),
    ])
}

pub fn chunks_schema(dim: usize) -> Schema {
    Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("chunk_idx", DataType::UInt32, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("normalized", DataType::Utf8, false),
        Field::new("description", DataType::Utf8, false),
        Field::new("chunk_type", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true, // nullable — None until embedded
        ),
        Field::new(
            "doc_embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        ),
        Field::new("materialization_tier", DataType::UInt8, false),
    ])
}

pub fn code_edges_schema() -> Schema {
    Schema::new(vec![
        Field::new("from_file", DataType::Utf8, false),
        Field::new("from_chunk", DataType::UInt32, false),
        Field::new("to_file", DataType::Utf8, false),
        Field::new("edge_type", DataType::Utf8, false),
    ])
}

pub fn call_edges_schema() -> Schema {
    Schema::new(vec![
        Field::new("caller_file", DataType::Utf8, false),
        Field::new("caller_symbol", DataType::Utf8, false),
        Field::new("callee_name", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("callee_file", DataType::Utf8, true),
        Field::new("callee_symbol", DataType::Utf8, true),
        Field::new("confidence", DataType::Float64, false),
        Field::new("dynamic", DataType::Boolean, false),
    ])
}

pub fn symbols_schema() -> Schema {
    Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
    ])
}

pub fn cochange_edges_schema() -> Schema {
    Schema::new(vec![
        Field::new("file_a", DataType::Utf8, false),
        Field::new("file_b", DataType::Utf8, false),
        // Only the Jaccard score and count are persisted.
        Field::new("cochange_count", DataType::UInt64, false),
        Field::new("jaccard", DataType::Float64, false),
    ])
}

pub fn sparse_index_schema() -> Schema {
    Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("chunk_idx", DataType::UInt32, false),
        Field::new("token_id", DataType::UInt32, false),
        Field::new("weight", DataType::Float32, false),
    ])
}

pub fn pagerank_scores_schema() -> Schema {
    Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
    ])
}

pub fn symbol_roles_schema() -> Schema {
    Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("role", DataType::Utf8, false),
    ])
}

/// Schema for the doc-embedding table (separate from chunks to avoid
/// wide rows when dual_embedding is disabled).
pub fn doc_chunks_schema(dim: usize) -> Schema {
    Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("chunk_idx", DataType::UInt32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ])
}
