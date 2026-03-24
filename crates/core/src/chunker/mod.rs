pub mod languages;

use languages::config_for_extension;
use text_splitter::{CodeSplitter, TextSplitter};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Query, QueryCursor};

/// The maximum non-whitespace character budget per chunk (Tier 1 and Tier 2).
const CHUNK_BUDGET: usize = 1500;

/// Minimum non-whitespace character count for a chunk to stand alone.
/// Chunks below this threshold are merge candidates.
const MIN_CHUNK_CHARS: usize = 40;

/// Minimum line span for a chunk to stand alone.
/// A chunk spanning fewer lines than this is also a merge candidate.
const MIN_CHUNK_LINES: usize = 2;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedChunk {
    pub chunk_idx: usize,
    pub content: String,
    /// Whitespace-normalised identifier form used for FTS indexing.
    pub normalized: String,
    pub chunk_type: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportEdge {
    pub from_file: String,
    pub to_file: String,
}

// ---------------------------------------------------------------------------
// Chunker
// ---------------------------------------------------------------------------

/// AST-aware chunker.
///
/// Tier 1 languages (Rust, Nix, Python, TypeScript, JavaScript, Go) use
/// tree-sitter `CodeSplitter` with a 1,500 non-whitespace character budget.
/// All other extensions fall back to a Tier 2 sliding-window `TextSplitter`.
pub struct Chunker;

impl Default for Chunker {
    fn default() -> Self {
        Self
    }
}

impl Chunker {
    /// Split `source` into chunks.  `rel_path` is the project-relative file
    /// path; it is used both for extension detection and as a prefix in each
    /// chunk's `normalized` field so BM25 FTS queries against file paths work.
    pub fn chunk_file(&self, rel_path: &str, source: &str) -> anyhow::Result<Vec<ParsedChunk>> {
        let ext = extension_of(rel_path);
        match config_for_extension(ext) {
            Some(cfg) => self.chunk_tier1(cfg.as_ref(), rel_path, source),
            None => self.chunk_tier2(rel_path, source),
        }
    }

    /// Extract import edges from `source`.  Returns an empty list for unknown
    /// extensions (Tier 2 files do not have parseable import syntax).
    pub fn extract_edges(&self, filename: &str, source: &str) -> anyhow::Result<Vec<ImportEdge>> {
        let ext = extension_of(filename);
        let cfg = match config_for_extension(ext) {
            Some(c) => c,
            None => return Ok(vec![]),
        };

        let language = cfg.language();
        let query_str = cfg.import_query();

        // Build tree.
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&language)
            .map_err(|e| anyhow::anyhow!("tree-sitter language set: {e}"))?;
        let tree = parser
            .parse(source.as_bytes(), None)
            .ok_or_else(|| anyhow::anyhow!("tree-sitter parse returned None for {filename}"))?;

        // Build query.  A malformed query is a programming error, not user
        // error — panic in debug, return empty in release.
        let query = match Query::new(&language, query_str) {
            Ok(q) => q,
            Err(e) => {
                if cfg!(debug_assertions) {
                    anyhow::bail!("bad import query for {filename}: {e}");
                } else {
                    return Ok(vec![]);
                }
            }
        };

        let mut cursor = QueryCursor::new();
        let source_bytes = source.as_bytes();
        let mut edges = Vec::new();

        // tree-sitter 0.24: QueryMatches does not implement Iterator; call next() directly.
        let mut matches = cursor.matches(&query, tree.root_node(), source_bytes);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let name = &query.capture_names()[cap.index as usize];
                if *name == "path" {
                    if let Ok(text) = cap.node.utf8_text(source_bytes) {
                        let text: &str = text;
                        // Strip surrounding quotes (' " ` ) if present.
                        let stripped = text.trim_matches(|c: char| c == '"' || c == '\'' || c == '`');
                        if !stripped.is_empty() {
                            edges.push(ImportEdge {
                                from_file: filename.to_string(),
                                to_file: stripped.to_string(),
                            });
                        }
                    }
                }
            }
        }

        Ok(edges)
    }

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    fn chunk_tier1(
        &self,
        cfg: &dyn languages::LanguageConfig,
        rel_path: &str,
        source: &str,
    ) -> anyhow::Result<Vec<ParsedChunk>> {
        let language = cfg.language();
        let splitter = CodeSplitter::new(language, CHUNK_BUDGET)
            .map_err(|e| anyhow::anyhow!("CodeSplitter init: {e}"))?;

        let raw_chunks: Vec<ParsedChunk> = splitter
            .chunks(source)
            .enumerate()
            .map(|(idx, chunk_text)| {
                let (start_line, end_line) = line_range_of(source, chunk_text);
                ParsedChunk {
                    chunk_idx: idx,
                    content: chunk_text.to_string(),
                    normalized: format!("{rel_path} code {}", normalize_for_fts(chunk_text)),
                    chunk_type: "code".to_string(),
                    start_line,
                    end_line,
                }
            })
            .collect();
        let chunks = merge_small_chunks(raw_chunks, rel_path);

        // If tree-sitter produced no splits (e.g. very short file), return the
        // whole source as a single chunk so callers always get at least one.
        if chunks.is_empty() && !source.trim().is_empty() {
            let line_count = source.lines().count();
            return Ok(vec![ParsedChunk {
                chunk_idx: 0,
                content: source.to_string(),
                normalized: format!("{rel_path} code {}", normalize_for_fts(source)),
                chunk_type: "code".to_string(),
                start_line: 1,
                end_line: line_count,
            }]);
        }

        Ok(chunks)
    }

    fn chunk_tier2(&self, rel_path: &str, source: &str) -> anyhow::Result<Vec<ParsedChunk>> {
        let splitter = TextSplitter::new(CHUNK_BUDGET);
        let raw_chunks: Vec<ParsedChunk> = splitter
            .chunks(source)
            .enumerate()
            .map(|(idx, chunk_text)| {
                let (start_line, end_line) = line_range_of(source, chunk_text);
                ParsedChunk {
                    chunk_idx: idx,
                    content: chunk_text.to_string(),
                    normalized: format!("{rel_path} text {}", normalize_for_fts(chunk_text)),
                    chunk_type: "text".to_string(),
                    start_line,
                    end_line,
                }
            })
            .collect();
        let chunks = merge_small_chunks(raw_chunks, rel_path);

        if chunks.is_empty() && !source.trim().is_empty() {
            let line_count = source.lines().count();
            return Ok(vec![ParsedChunk {
                chunk_idx: 0,
                content: source.to_string(),
                normalized: format!("{rel_path} text {}", normalize_for_fts(source)),
                chunk_type: "text".to_string(),
                start_line: 1,
                end_line: line_count,
            }]);
        }

        Ok(chunks)
    }
}

// ---------------------------------------------------------------------------
// merge_small_chunks
// ---------------------------------------------------------------------------

/// Returns true when `chunk` is too small to stand alone as a search result.
#[inline]
fn chunk_is_tiny(chunk: &ParsedChunk) -> bool {
    chunk.content.trim().len() < MIN_CHUNK_CHARS
        && (chunk.end_line.saturating_sub(chunk.start_line) + 1) < MIN_CHUNK_LINES
}

/// Merge chunks that are too small to be semantically useful into a neighbor,
/// then re-number the survivors.
///
/// **Pass 1 (backward):** a tiny chunk that has a predecessor is appended to
/// that predecessor.  Handles trailing noise such as a lone `}` or
/// `struct Foo;` that appears *after* a larger declaration.
///
/// **Pass 2 (forward):** if the very first chunk is still tiny and a
/// successor exists, it is prepended to that successor.  Handles leading
/// one-liners like `struct Foo;` that appear *before* an impl block.
fn merge_small_chunks(chunks: Vec<ParsedChunk>, rel_path: &str) -> Vec<ParsedChunk> {
    if chunks.len() <= 1 {
        return chunks;
    }

    // Pass 1: merge tiny chunks into their predecessor.
    let mut merged: Vec<ParsedChunk> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        if chunk_is_tiny(&chunk) && !merged.is_empty() {
            let prev = merged.last_mut().unwrap();
            prev.content.push('\n');
            prev.content.push_str(&chunk.content);
            prev.end_line = chunk.end_line;
            prev.normalized = format!("{rel_path} {} {}", prev.chunk_type, normalize_for_fts(&prev.content));
        } else {
            merged.push(chunk);
        }
    }

    // Pass 2: if the leading chunk is still tiny and a successor exists,
    // prepend it to that successor.
    if merged.len() >= 2 && chunk_is_tiny(&merged[0]) {
        let first = merged.remove(0); // O(n) but chunk counts are tiny
        let second = &mut merged[0];
        let mut new_content = first.content;
        new_content.push('\n');
        new_content.push_str(&second.content);
        second.content = new_content;
        second.start_line = first.start_line;
        second.normalized = format!(
            "{rel_path} {} {}",
            second.chunk_type,
            normalize_for_fts(&second.content)
        );
    }

    // Re-number survivors.
    for (i, chunk) in merged.iter_mut().enumerate() {
        chunk.chunk_idx = i;
    }

    merged
}

// ---------------------------------------------------------------------------
// normalize_for_fts
// ---------------------------------------------------------------------------

/// Convert a code identifier or snippet to a space-separated lowercase word
/// sequence suitable for FTS indexing.
///
/// Rules:
/// - Split on any non-alphanumeric character (underscores, hyphens, dots, …).
/// - Within each resulting segment, split camelCase and PascalCase boundaries.
/// - Consecutive-uppercase runs (acronyms like `HTTP`) are kept together until
///   the transition back to lowercase: `parseHTTPResponse` → `parse http response`.
/// - All output is lowercased.
///
/// # Example
/// ```
/// use skelesearch_core::chunker::normalize_for_fts;
/// assert_eq!(normalize_for_fts("parseHTTPResponse_json"), "parse http response json parsehttp httpresponse responsejson");
/// ```
pub fn normalize_for_fts(text: &str) -> String {
    let mut words: Vec<String> = Vec::new();
    for segment in text.split(|c: char| !c.is_alphanumeric()) {
        if segment.is_empty() {
            continue;
        }
        split_camel(segment, &mut words);
    }

    // Generate contiguous bigrams for partial identifier matching.
    // e.g., ["get", "user", "by", "id"] → also adds "getuser", "userby", "byid"
    // This lets BM25 match queries like "userById" against an identifier
    // that was split into separate words.
    let bigrams: Vec<String> = words.windows(2)
        .map(|w| format!("{}{}", w[0], w[1]))
        .collect();

    let mut result = words.join(" ");
    if !bigrams.is_empty() {
        result.push(' ');
        result.push_str(&bigrams.join(" "));
    }
    result
}

/// Split a single alphanumeric segment at camelCase / PascalCase / acronym
/// boundaries and push lowercase words into `out`.
fn split_camel(s: &str, out: &mut Vec<String>) {
    // Only operate on ASCII to keep indexing simple and predictable.
    // Non-ASCII segments are emitted as a single lowercase word.
    if !s.is_ascii() {
        out.push(s.to_lowercase());
        return;
    }

    let b = s.as_bytes();
    let n = b.len();
    let mut start = 0usize;

    for i in 1..n {
        let prev = b[i - 1];
        let curr = b[i];
        let next = b.get(i + 1).copied();

        let split = curr.is_ascii_uppercase()
            && (prev.is_ascii_lowercase()
                || (prev.is_ascii_uppercase()
                    && next.map(|c| c.is_ascii_lowercase()).unwrap_or(false)));

        if split {
            // Safety: start..i is a valid byte range in an ASCII string.
            out.push(s[start..i].to_lowercase());
            start = i;
        }
    }
    if start < n {
        out.push(s[start..].to_lowercase());
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the file extension (without the dot) for `filename`, or `""`.
fn extension_of(filename: &str) -> &str {
    std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
}

/// Compute the 1-based start/end line numbers of `chunk` within `source`.
///
/// `chunk` must be a sub-slice of `source` (as returned by text-splitter);
/// pointer arithmetic is used to avoid a potentially-ambiguous string search.
fn line_range_of(source: &str, chunk: &str) -> (usize, usize) {
    let source_start = source.as_ptr() as usize;
    let chunk_start = chunk.as_ptr() as usize;

    // If the pointers are out of range (shouldn't happen with text-splitter),
    // fall back to 1-based single-line.
    if chunk_start < source_start || chunk_start > source_start + source.len() {
        return (1, 1 + chunk.bytes().filter(|&b| b == b'\n').count());
    }

    let byte_start = chunk_start - source_start;
    let byte_end = byte_start + chunk.len();

    let start_line = source[..byte_start].bytes().filter(|&b| b == b'\n').count() + 1;
    let end_line = source[..byte_end].bytes().filter(|&b| b == b'\n').count() + 1;

    (start_line, end_line)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_camel_case() {
        assert_eq!(normalize_for_fts("parseHTTPResponse_json"), "parse http response json parsehttp httpresponse responsejson");
    }

    #[test]
    fn normalize_pascal() {
        assert_eq!(normalize_for_fts("MyStruct"), "my struct mystruct");
    }

    #[test]
    fn normalize_snake_case() {
        assert_eq!(normalize_for_fts("foo_bar_baz"), "foo bar baz foobar barbaz");
    }


    // ------------------------------------------------------------------
    // merge_small_chunks
    // ------------------------------------------------------------------

    /// Verify that a trivially small leading declaration is merged into its
    /// successor rather than emitted as a standalone chunk.
    ///
    /// The source is a valid Rust file whose first AST node (`struct Tiny;`)
    /// is a one-liner with far fewer non-whitespace chars than MIN_CHUNK_CHARS,
    /// followed by an impl block whose body is padded large enough to force
    /// CodeSplitter to begin a new chunk (> CHUNK_BUDGET non-ws chars).
    #[test]
    fn tiny_struct_merges_into_impl_chunk() {
        // Each line in the padding block is ~50 chars of non-whitespace.
        // 32 lines × 50 ≈ 1600 non-ws chars, which exceeds CHUNK_BUDGET (1500),
        // so CodeSplitter should split struct Tiny; into its own raw chunk
        // before the impl body.
        let padding: String = (0..32)
            .map(|i| format!("    fn pad_{i:02}(&self) -> u32 {{ {i} + {i} }}\n"))
            .collect();
        let source = format!(
            "struct Tiny;\n\nimpl Tiny {{\n{padding}}}\n"
        );

        let chunker = Chunker::default();
        let chunks = chunker
            .chunk_file("src/lib.rs", &source)
            .expect("chunk_file should not fail on valid Rust");

        // After merging there must be at least one chunk.
        assert!(!chunks.is_empty(), "expected at least one chunk");

        // 'struct Tiny;' must not appear as a standalone chunk; it must have
        // been merged into its neighbor.
        let standalone = chunks.iter().any(|c| {
            let t = c.content.trim();
            t == "struct Tiny;" || t == "struct Tiny ;"
        });
        assert!(
            !standalone,
            "struct Tiny; should have been merged, but appears alone in: {chunks:#?}"
        );

        // The merged chunk that absorbed 'struct Tiny;' must contain it.
        let contains_struct = chunks.iter().any(|c| c.content.contains("struct Tiny;"));
        assert!(
            contains_struct,
            "no chunk contains 'struct Tiny;' after merge: {chunks:#?}"
        );

        // chunk_idx values must be a contiguous 0..n sequence.
        for (expected, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                chunk.chunk_idx, expected,
                "chunk_idx out of order at position {expected}: {chunk:#?}"
            );
        }
    }

    /// Verify that a trivially small trailing fragment is merged into its
    /// predecessor rather than emitted as a standalone chunk.
    #[test]
    fn tiny_trailing_fragment_merges_into_predecessor() {
        // Build a large leading function then a tiny one-liner declaration.
        let padding: String = (0..32)
            .map(|i| format!("    let _{i:02} = {i} * {i};\n"))
            .collect();
        let source = format!(
            "fn large() {{\n{padding}}}\n\nstruct Tail;\n"
        );

        let chunker = Chunker::default();
        let chunks = chunker
            .chunk_file("src/lib.rs", &source)
            .expect("chunk_file should not fail on valid Rust");

        assert!(!chunks.is_empty());

        let standalone = chunks.iter().any(|c| {
            let t = c.content.trim();
            t == "struct Tail;" || t == "struct Tail ;"
        });
        assert!(!standalone, "struct Tail; should have been merged: {chunks:#?}");

        let contains_tail = chunks.iter().any(|c| c.content.contains("struct Tail;"));
        assert!(contains_tail, "no chunk contains 'struct Tail;': {chunks:#?}");
    }
}
