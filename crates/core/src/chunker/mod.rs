pub mod languages;

use languages::config_for_extension;
use text_splitter::{CodeSplitter, TextSplitter};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Query, QueryCursor};

/// The maximum non-whitespace character budget per chunk (Tier 1 and Tier 2).
const CHUNK_BUDGET: usize = 1500;

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

        let chunks: Vec<ParsedChunk> = splitter
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
        let chunks: Vec<ParsedChunk> = splitter
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
}
