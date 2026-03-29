/// Tantivy full-text search index with a custom code-aware tokenizer.
///
/// The `CodeAnalyzer` tokenizer splits identifiers on:
/// - CamelCase word boundaries
/// - snake_case / kebab-case separators
/// - Numeric segments
///
/// This resolves the CozoDB limitation where `getUserById` would not match
/// queries for `user`, `id`, or `get` (ADR-011).
use std::path::Path;
use std::sync::{Arc, Mutex};
use tantivy::schema::{Field, Schema, SchemaBuilder, TextFieldIndexing, TextOptions, FAST, STORED};
use tantivy::tokenizer::{
    Language, LowerCaser, RegexTokenizer, SimpleTokenizer, Stemmer, TextAnalyzer,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy};

const CODE_ANALYZER_NAME: &str = "code";

/// Code-aware text analyzer.
///
/// Splits on CamelCase, snake_case, kebab-case, and numeric boundaries so
/// `getUserById` tokenizes to ["get", "user", "by", "id"].
fn build_code_analyzer() -> TextAnalyzer {
    // Splits identifiers on CamelCase and non-alphanumeric boundaries.
    // Note: the regex crate does NOT support lookaheads — `(?=...)` is not valid.
    //   [A-Z]?[a-z]+ — lowercase word (camelCase inner words, e.g. `user` in `getUserById`)
    //   [A-Z]+        — consecutive uppercase run (e.g. `HTTP`)
    //   [0-9]+        — numeric segment
    // snake_case and kebab-case are implicitly split because `_` and `-` are not matched.
    let token_regex = r"[A-Z]?[a-z]+|[A-Z]+|[0-9]+";

    TextAnalyzer::builder(RegexTokenizer::new(token_regex).expect("valid regex"))
        .filter(LowerCaser)
        .build()
}

/// Natural-language analyzer for LLM summaries (description field).
fn build_nl_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(Stemmer::new(Language::English))
        .build()
}

/// Tantivy index wrapper with typed field handles and a shared `IndexWriter`.
pub struct TantivyIndex {
    pub index: Index,
    pub reader: IndexReader,
    pub writer: Option<Arc<Mutex<IndexWriter>>>,
    pub schema: Schema,
    // Typed field handles for building documents without string lookups.
    pub f_file_path: Field,
    pub f_chunk_idx: Field,
    pub f_normalized: Field,
    pub f_description: Field,
    pub f_chunk_type: Field,
    pub f_tier: Field,
}

impl TantivyIndex {
    /// Open or create a Tantivy index at `path`.
    ///
    /// Registers the `CodeAnalyzer` tokenizer on first call and on every
    /// subsequent open (idempotent — Tantivy allows re-registering with the
    /// same name).
    pub fn open_or_create(path: &Path) -> anyhow::Result<Self> {
        Self::open_or_create_with_mode(path, true)
    }

    /// Open or create a Tantivy index without acquiring a writer lock.
    pub fn open_or_create_read_only(path: &Path) -> anyhow::Result<Self> {
        Self::open_or_create_with_mode(path, false)
    }

    fn open_or_create_with_mode(path: &Path, writable: bool) -> anyhow::Result<Self> {
        std::fs::create_dir_all(path)?;

        // Build schema.
        let mut builder = SchemaBuilder::new();

        // file_path: stored, fast (for term deletion queries), raw tokenizer.
        let f_file_path = builder.add_text_field("file_path", STORED | FAST);

        // chunk_idx: stored, fast (u64 for efficient deletion filter).
        let f_chunk_idx = builder.add_u64_field("chunk_idx", STORED | FAST);

        // normalized: tokenized with CodeAnalyzer for BM25.
        let code_text_opts = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(CODE_ANALYZER_NAME)
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        );
        let f_normalized = builder.add_text_field("normalized", code_text_opts);

        // description: tokenized with natural-language analyzer (stemmed).
        let nl_text_opts = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("en_stem")
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        );
        let f_description = builder.add_text_field("description", nl_text_opts);

        // chunk_type: stored only (not searchable, used for filtering in Rust).
        let f_chunk_type = builder.add_text_field("chunk_type", STORED);

        // tier: stored + fast for efficient tier1 deletion.
        let f_tier = builder.add_u64_field("tier", STORED | FAST);

        let schema = builder.build();

        // Open or create index.
        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(path)?
        } else {
            Index::create_in_dir(path, schema.clone())?
        };

        // Register tokenizers (idempotent).
        index
            .tokenizers()
            .register(CODE_ANALYZER_NAME, build_code_analyzer());
        index.tokenizers().register("en_stem", build_nl_analyzer());

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        let writer = if writable {
            // 50 MB heap for the IndexWriter — adequate for batch indexing.
            Some(Arc::new(Mutex::new(index.writer(50_000_000)?)))
        } else {
            None
        };

        Ok(Self {
            index,
            reader,
            writer,
            schema,
            f_file_path,
            f_chunk_idx,
            f_normalized,
            f_description,
            f_chunk_type,
            f_tier,
        })
    }
}
