use crate::chunker::languages::config_for_extension;
use tree_sitter::Language;
use tree_sitter_tags::{TagsConfiguration, TagsContext};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SymbolDef {
    pub file_path: String,
    pub name: String,
    /// Normalised kind: "function", "struct", "class", "method", "trait",
    /// "enum", "type", "impl", "interface", or the raw tree-sitter node kind
    /// for anything not explicitly mapped.
    pub kind: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// A `@reference.call` capture: a function or method call site within a file.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceCapture {
    pub name: String,
    pub start_line: usize,
    pub end_line: usize,
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// Extract named symbol definitions from `source` by parsing it with the
/// tree-sitter grammar inferred from `filename`'s extension.
///
/// Attempts to use tree-sitter-tags for richer extraction; falls back to the
/// manual AST traversal when no tags query is available for the language or
/// when `TagsConfiguration` fails to compile.
///
/// Returns an empty `Vec` for unknown extensions or source that yields no
/// top-level named nodes — not an error.
pub fn extract_symbols(filename: &str, source: &str) -> anyhow::Result<Vec<SymbolDef>> {
    let ext = std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // Unknown extension: spec says return empty, not an error.
    let cfg = match config_for_extension(ext) {
        Some(c) => c,
        None => return Ok(vec![]),
    };

    // Try the tree-sitter-tags path first.
    if let Some((tags_query, lang)) = tags_info_for_extension(ext) {
        match TagsConfiguration::new(lang, tags_query, "") {
            Ok(tags_cfg) => {
                return extract_via_tags(&tags_cfg, source, filename);
            }
            Err(e) => {
                // The tags query failed to compile for this language version;
                // log at debug and fall through to the manual path.
                tracing::debug!(
                    ext,
                    error = %e,
                    "tree-sitter-tags config failed; falling back to manual traversal"
                );
            }
        }
    }

    // Manual fallback: walk the AST looking for chunk-level nodes.
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&cfg.language())
        .map_err(|e| anyhow::anyhow!("set_language: {e}"))?;

    let tree = parser
        .parse(source.as_bytes(), None)
        .ok_or_else(|| anyhow::anyhow!("parse failed for {filename}"))?;

    let mut symbols = Vec::new();
    collect_symbols(
        &tree.root_node(),
        source,
        filename,
        cfg.chunk_node_kinds(),
        &mut symbols,
    );
    Ok(symbols)
}

/// Extract `@reference.call` captures from `source` using the tree-sitter-tags
/// pipeline for the language inferred from `filename`'s extension.
///
/// Returns an empty `Vec` when the language has no tags query, when the query
/// fails to compile, or when `source` yields no call references — not an error.
pub fn extract_references(filename: &str, source: &str) -> anyhow::Result<Vec<ReferenceCapture>> {
    let ext = std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let (tags_query, lang) = match tags_info_for_extension(ext) {
        Some(info) => info,
        None => return Ok(vec![]),
    };

    match TagsConfiguration::new(lang, tags_query, "") {
        Ok(tags_cfg) => extract_references_via_tags(&tags_cfg, source),
        Err(e) => {
            // Same graceful degradation as extract_symbols.
            tracing::debug!(
                ext,
                error = %e,
                "tree-sitter-tags config failed; skipping reference extraction"
            );
            Ok(vec![])
        }
    }
}

fn extract_references_via_tags(
    config: &TagsConfiguration,
    source: &str,
) -> anyhow::Result<Vec<ReferenceCapture>> {
    let mut ctx = TagsContext::new();
    let source_bytes = source.as_bytes();

    let (iter, _cancelled) = ctx
        .generate_tags(config, source_bytes, None)
        .map_err(|e| anyhow::anyhow!("generate_tags: {e}"))?;

    let mut refs = Vec::new();
    for result in iter {
        let tag = match result {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!(error = %e, "skipping tag error in reference extraction");
                continue;
            }
        };

        // Only emit references (not definitions).
        if tag.is_definition {
            continue;
        }

        // Filter to call references specifically (@reference.call).
        let syntax_type = config.syntax_type_name(tag.syntax_type_id);
        if syntax_type != "call" {
            continue;
        }

        // Skip tags with empty name ranges (anonymous constructs).
        if tag.name_range.is_empty() {
            continue;
        }

        let name = match std::str::from_utf8(&source_bytes[tag.name_range.clone()]) {
            Ok(s) if !s.is_empty() => s.to_string(),
            _ => continue,
        };

        refs.push(ReferenceCapture {
            name,
            start_line: tag.span.start.row + 1, // 0-indexed → 1-indexed
            end_line: tag.span.end.row + 1,
        });
    }

    Ok(refs)
}

// ---------------------------------------------------------------------------
// tree-sitter-tags extraction
// ---------------------------------------------------------------------------

/// Returns the tags query string and the correct `Language` for `ext`, or
/// `None` if the language crate does not expose a `TAGS_QUERY` constant.
///
/// Note: tsx gets `LANGUAGE_TSX` (not `LANGUAGE_TYPESCRIPT`) so the correct
/// grammar is used; both share the same `tags.scm`.
/// Languages without a committed tags query (c-sharp, kotlin-sg, scala, nix)
/// return `None` and fall through to the manual traversal.
fn tags_info_for_extension(ext: &str) -> Option<(&'static str, Language)> {
    match ext {
        "rs" => Some((
            tree_sitter_rust::TAGS_QUERY,
            tree_sitter_rust::LANGUAGE.into(),
        )),
        "py" => Some((
            tree_sitter_python::TAGS_QUERY,
            tree_sitter_python::LANGUAGE.into(),
        )),
        // ts and tsx share the same tags.scm but use different grammars.
        "ts" => Some((
            tree_sitter_typescript::TAGS_QUERY,
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        )),
        "tsx" => Some((
            tree_sitter_typescript::TAGS_QUERY,
            tree_sitter_typescript::LANGUAGE_TSX.into(),
        )),
        "js" | "jsx" | "mjs" | "cjs" => Some((
            tree_sitter_javascript::TAGS_QUERY,
            tree_sitter_javascript::LANGUAGE.into(),
        )),
        "go" => Some((
            tree_sitter_go::TAGS_QUERY,
            tree_sitter_go::LANGUAGE.into(),
        )),
        "java" => Some((
            tree_sitter_java::TAGS_QUERY,
            tree_sitter_java::LANGUAGE.into(),
        )),
        "c" | "h" => Some((
            tree_sitter_c::TAGS_QUERY,
            tree_sitter_c::LANGUAGE.into(),
        )),
        "cpp" | "cc" | "cxx" | "hpp" => Some((
            tree_sitter_cpp::TAGS_QUERY,
            tree_sitter_cpp::LANGUAGE.into(),
        )),
        "rb" => Some((
            tree_sitter_ruby::TAGS_QUERY,
            tree_sitter_ruby::LANGUAGE.into(),
        )),
        "php" => Some((
            tree_sitter_php::TAGS_QUERY,
            tree_sitter_php::LANGUAGE_PHP.into(),
        )),
        "swift" => Some((
            tree_sitter_swift::TAGS_QUERY,
            tree_sitter_swift::LANGUAGE.into(),
        )),
        // c-sharp: TAGS_QUERY is commented out in the crate — no tags query.
        // kotlin-sg: TAGS_QUERY is commented out in the crate — no tags query.
        // scala: no TAGS_QUERY in the crate.
        // nix: TAGS_QUERY is commented out in the crate — no tags query.
        _ => None,
    }
}

fn extract_via_tags(
    config: &TagsConfiguration,
    source: &str,
    filename: &str,
) -> anyhow::Result<Vec<SymbolDef>> {
    let mut ctx = TagsContext::new();
    let source_bytes = source.as_bytes();

    // generate_tags returns (iterator, cancelled_bool). We pass no cancellation
    // flag, so the bool is always false; ignore it.
    let (iter, _cancelled) = ctx
        .generate_tags(config, source_bytes, None)
        .map_err(|e| anyhow::anyhow!("generate_tags: {e}"))?;

    let mut symbols = Vec::new();
    for result in iter {
        let tag = match result {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!(error = %e, "skipping tag error");
                continue;
            }
        };

        // Only emit definitions; skip references.
        if !tag.is_definition {
            continue;
        }

        // Skip tags with empty name ranges (anonymous constructs).
        if tag.name_range.is_empty() {
            continue;
        }

        // Extract the name from the byte range into source.
        let name = match std::str::from_utf8(&source_bytes[tag.name_range.clone()]) {
            Ok(s) if !s.is_empty() => s.to_string(),
            _ => continue,
        };

        // syntax_type_name already strips the "definition." prefix — it returns
        // just the suffix: "function", "class", "method", etc.
        let kind = config.syntax_type_name(tag.syntax_type_id).to_string();

        symbols.push(SymbolDef {
            file_path: filename.to_string(),
            name,
            kind,
            start_line: tag.span.start.row + 1, // 0-indexed → 1-indexed
            end_line: tag.span.end.row + 1,
        });
    }

    Ok(symbols)
}

// ---------------------------------------------------------------------------
// Fallback: manual AST traversal
// ---------------------------------------------------------------------------

fn collect_symbols(
    node: &tree_sitter::Node,
    source: &str,
    filename: &str,
    kinds: &[&str],
    out: &mut Vec<SymbolDef>,
) {
    if kinds.contains(&node.kind()) {
        // Only emit nodes that have an explicit `name` field; nodes without
        // one (e.g. anonymous Rust `impl` blocks) are skipped per spec.
        if let Some(name_node) = node.child_by_field_name("name") {
            if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                out.push(SymbolDef {
                    file_path: filename.to_string(),
                    name: name.to_string(),
                    kind: normalize_kind(node.kind()),
                    start_line: node.start_position().row + 1, // 0-indexed → 1-indexed
                    end_line: node.end_position().row + 1,
                });
            }
        }
    }

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            collect_symbols(&child, source, filename, kinds, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Kind normalisation (fallback path only)
// ---------------------------------------------------------------------------

/// Map raw tree-sitter node kinds to a small, stable vocabulary.
/// Anything not in this table is passed through unchanged so callers always
/// get _something_ rather than an empty string.
fn normalize_kind(kind: &str) -> String {
    match kind {
        "function_item" | "function_definition" | "function_declaration" => "function",
        "method_declaration" | "method_definition" => "method",
        "struct_item" => "struct",
        "trait_item" => "trait",
        "enum_item" => "enum",
        "class_declaration" | "class_definition" => "class",
        "impl_item" => "impl",
        "interface_declaration" => "interface",
        "type_declaration" => "type",
        other => other,
    }
    .to_string()
}
