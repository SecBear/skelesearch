use crate::chunker::languages::config_for_extension;
use tree_sitter::Parser;

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

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// Extract named symbol definitions from `source` by parsing it with the
/// tree-sitter grammar inferred from `filename`'s extension.
///
/// Returns an empty `Vec` for unknown extensions or source that yields no
/// top-level named nodes — not an error.  Only nodes that carry a `name`
/// field in the grammar are emitted; anonymous nodes (e.g. bare `impl`
/// blocks with no associated name field) are silently skipped per spec.
pub fn extract_symbols(filename: &str, source: &str) -> anyhow::Result<Vec<SymbolDef>> {
    let ext = std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let cfg = match config_for_extension(ext) {
        Some(c) => c,
        // Unknown extension: spec says return empty, not an error.
        None => return Ok(vec![]),
    };

    let mut parser = Parser::new();
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

// ---------------------------------------------------------------------------
// AST traversal
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
// Kind normalisation
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
