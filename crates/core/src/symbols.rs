use crate::chunker::languages::config_for_extension;
use tree_sitter::Language;
use tree_sitter_tags::{TagsConfiguration, TagsContext};

// ---------------------------------------------------------------------------
// Custom tags queries
// ---------------------------------------------------------------------------

/// TypeScript's built-in TAGS_QUERY only captures `@reference.type` and
/// `@reference.class` — it has no `@reference.call` patterns, so call-graph
/// edges are entirely absent for `.ts`/`.tsx` files.
///
/// This custom query keeps every definition pattern from the upstream
/// `tree_sitter_typescript::TAGS_QUERY` verbatim, then appends
/// `@reference.call` captures modelled after the JavaScript tags query.
/// TypeScript inherits `call_expression` from the JavaScript grammar, so
/// the same node shapes apply.
///
/// Convention: `@name` sub-capture = identifier text; `@reference.call`
/// on the enclosing call / arguments node = the tag site.
const TS_CUSTOM_TAGS_QUERY: &str = r#"
; Original TypeScript definitions (verbatim from built-in TAGS_QUERY).
(function_signature
  name: (identifier) @name) @definition.function

(method_signature
  name: (property_identifier) @name) @definition.method

(abstract_method_signature
  name: (property_identifier) @name) @definition.method

(abstract_class_declaration
  name: (type_identifier) @name) @definition.class

(module
  name: (identifier) @name) @definition.module

(interface_declaration
  name: (type_identifier) @name) @definition.interface

(type_annotation
  (type_identifier) @name) @reference.type

(new_expression
  constructor: (identifier) @name) @reference.class

; Added: call-site references.
; Direct call: foo() -- excludes require() (module boundary, not a semantic edge).
(
  (call_expression
    function: (identifier) @name) @reference.call
  (#not-match? @name "^(require)$")
)

; Member call: obj.method() -- @name = property identifier; @reference.call = arguments node.
(call_expression
  function: (member_expression
    property: (property_identifier) @name)
  arguments: (_) @reference.call)
"#;

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

/// An alias introduced by an import statement.
///
/// For `use crate::schema as s`, `alias = "s"`, `original = "schema"`.
/// For `from os import path as p`, `alias = "p"`, `original = "path"`,
/// `source_path = Some("os")`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportAlias {
    /// The alias name used in code (e.g. `s` from `use crate::schema as s`).
    pub alias: String,
    /// The original name being aliased (last path segment, e.g. `schema`).
    pub original: String,
    /// The source module/path, if available (e.g. `os` from `from os import path as p`).
    pub source_path: Option<String>,
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
        // ts and tsx share the same grammar shape but the built-in TAGS_QUERY
        // has no @reference.call patterns; use TS_CUSTOM_TAGS_QUERY instead.
        "ts" => Some((
            TS_CUSTOM_TAGS_QUERY,
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        )),
        "tsx" => Some((
            TS_CUSTOM_TAGS_QUERY,
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


// ---------------------------------------------------------------------------
// Import alias extraction
// ---------------------------------------------------------------------------

/// Extract import aliases from `source` for the language inferred from `filename`.
///
/// Covers `use … as` (Rust), `import … as` / `from … import … as` (Python),
/// `import { x as y }` and `import * as ns` (TypeScript/JavaScript).
///
/// Returns an empty `Vec` for unknown extensions, unparseable source, or files
/// with no aliases — never an error.
pub fn extract_import_aliases(filename: &str, source: &str) -> Vec<ImportAlias> {
    let ext = std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let cfg = match config_for_extension(ext) {
        Some(c) => c,
        None => return vec![],
    };

    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&cfg.language()).is_err() {
        return vec![];
    }

    let tree = match parser.parse(source.as_bytes(), None) {
        Some(t) => t,
        None => return vec![],
    };

    let mut aliases = Vec::new();
    let src = source.as_bytes();

    match ext {
        "rs" => collect_rust_aliases(&tree.root_node(), src, &mut aliases),
        "py" => collect_python_aliases(&tree.root_node(), src, None, &mut aliases),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => {
            collect_ts_aliases(&tree.root_node(), src, None, &mut aliases);
        }
        _ => {}
    }

    aliases
}

// --- Rust alias helpers ---

fn collect_rust_aliases(node: &tree_sitter::Node, source: &[u8], out: &mut Vec<ImportAlias>) {
    if node.kind() == "use_as_clause" {
        if let Some(path_node) = node.child_by_field_name("path") {
            if let Some(alias_node) = node.child_by_field_name("alias") {
                let original = rust_path_last_segment(&path_node, source);
                if let Ok(alias_str) = alias_node.utf8_text(source) {
                    let alias_str = alias_str.trim();
                    // `_` is a non-binding discard — skip it.
                    if !alias_str.is_empty() && alias_str != "_" {
                        if let Some(orig) = original {
                            out.push(ImportAlias {
                                alias: alias_str.to_string(),
                                original: orig,
                                source_path: None,
                            });
                        }
                    }
                }
            }
        }
    }
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            collect_rust_aliases(&child, source, out);
        }
    }
}

/// Extract the last segment from a Rust path node.
///
/// `identifier` → its text; `scoped_identifier` → its `name` field (last segment).
fn rust_path_last_segment(node: &tree_sitter::Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "identifier" => node.utf8_text(source).ok().map(|s| s.trim().to_string()),
        "scoped_identifier" => node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source).ok())
            .map(|s| s.trim().to_string()),
        _ => {
            // Unknown shape: use last named child as fallback.
            let count = node.named_child_count();
            if count > 0 {
                node.named_child((count - 1) as u32)
                    .and_then(|n| n.utf8_text(source).ok())
                    .map(|s| s.trim().to_string())
            } else {
                node.utf8_text(source).ok().map(|s| s.trim().to_string())
            }
        }
    }
}

// --- Python alias helpers ---

fn collect_python_aliases(
    node: &tree_sitter::Node,
    source: &[u8],
    from_module: Option<&str>,
    out: &mut Vec<ImportAlias>,
) {
    match node.kind() {
        "import_from_statement" => {
            // Extract the module name so aliased children know their source.
            let module: Option<String> = node
                .child_by_field_name("module_name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(|s| s.trim().to_string());
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    collect_python_aliases(&child, source, module.as_deref(), out);
                }
            }
        }
        "aliased_import" => {
            if let (Some(name_node), Some(alias_node)) = (
                node.child_by_field_name("name"),
                node.child_by_field_name("alias"),
            ) {
                if let (Ok(original_text), Ok(alias_str)) = (
                    name_node.utf8_text(source),
                    alias_node.utf8_text(source),
                ) {
                    let alias_str = alias_str.trim();
                    if !alias_str.is_empty() && alias_str != "_" {
                        // dotted_name like `os.path` → take last segment `path`.
                        let original_stem = original_text
                            .trim()
                            .rsplit('.')
                            .next()
                            .unwrap_or(original_text.trim())
                            .to_string();
                        out.push(ImportAlias {
                            alias: alias_str.to_string(),
                            original: original_stem,
                            source_path: from_module.map(|s| s.to_string()),
                        });
                    }
                }
            }
        }
        _ => {
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    collect_python_aliases(&child, source, from_module, out);
                }
            }
        }
    }
}

// --- TypeScript / JavaScript alias helpers ---

fn collect_ts_aliases(
    node: &tree_sitter::Node,
    source: &[u8],
    from_source: Option<&str>,
    out: &mut Vec<ImportAlias>,
) {
    match node.kind() {
        "import_statement" => {
            // Locate the module string for context attachment.
            let src_path: Option<String> = find_ts_import_source(node, source);
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    collect_ts_aliases(&child, source, src_path.as_deref(), out);
                }
            }
        }
        "import_specifier" => {
            // `import { foo as bar }` — only emit when the alias field is present.
            if let (Some(name_node), Some(alias_node)) = (
                node.child_by_field_name("name"),
                node.child_by_field_name("alias"),
            ) {
                if let (Ok(original), Ok(alias_str)) = (
                    name_node.utf8_text(source),
                    alias_node.utf8_text(source),
                ) {
                    let alias_str = alias_str.trim();
                    if !alias_str.is_empty() && alias_str != "_" {
                        out.push(ImportAlias {
                            alias: alias_str.to_string(),
                            original: original.trim().to_string(),
                            source_path: from_source.map(|s| s.to_string()),
                        });
                    }
                }
            }
            // import_specifier has no nested alias nodes worth recursing into.
        }
        "namespace_import" => {
            // `import * as ns from "./utils"` — the identifier has no dedicated
            // field in the grammar; it is the first (only) named child.
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i as u32) {
                    if child.kind() == "identifier" {
                        if let Ok(alias_str) = child.utf8_text(source) {
                            let alias_str = alias_str.trim();
                            if !alias_str.is_empty() && alias_str != "_" {
                                // Use the stem of the import path as `original` so
                                // the alias can resolve via the stem-keyed import_targets.
                                let original = from_source
                                    .and_then(|p| {
                                        std::path::Path::new(p)
                                            .file_stem()
                                            .and_then(|s| s.to_str())
                                            .map(|s| s.to_string())
                                    })
                                    .unwrap_or_else(|| "*".to_string());
                                out.push(ImportAlias {
                                    alias: alias_str.to_string(),
                                    original,
                                    source_path: from_source.map(|s| s.to_string()),
                                });
                            }
                        }
                        break; // Only one identifier in namespace_import.
                    }
                }
            }
        }
        _ => {
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    collect_ts_aliases(&child, source, from_source, out);
                }
            }
        }
    }
}

/// Walk an `import_statement` node to find the string literal naming the module.
/// Returns the path with surrounding quote characters stripped.
fn find_ts_import_source(node: &tree_sitter::Node, source: &[u8]) -> Option<String> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "string" => {
                    if let Ok(s) = child.utf8_text(source) {
                        return Some(strip_import_quotes(s));
                    }
                }
                "from_clause" => {
                    for j in 0..child.child_count() {
                        if let Some(gc) = child.child(j as u32) {
                            if gc.kind() == "string" {
                                if let Ok(s) = gc.utf8_text(source) {
                                    return Some(strip_import_quotes(s));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Strip surrounding quote characters (`"`, `'`, `` ` ``) from an import path.
fn strip_import_quotes(s: &str) -> String {
    s.trim_matches(|c| c == '"' || c == '\'' || c == '`').to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod import_alias_tests {
    use super::{extract_import_aliases, ImportAlias};

    fn alias(alias: &str, original: &str) -> ImportAlias {
        ImportAlias { alias: alias.to_string(), original: original.to_string(), source_path: None }
    }

    fn alias_with_src(alias: &str, original: &str, src: &str) -> ImportAlias {
        ImportAlias {
            alias: alias.to_string(),
            original: original.to_string(),
            source_path: Some(src.to_string()),
        }
    }

    #[test]
    fn rust_simple_alias() {
        let src = "use crate::schema as s;\n";
        let result = extract_import_aliases("foo.rs", src);
        assert_eq!(result, vec![alias("s", "schema")]);
    }

    #[test]
    fn rust_grouped_aliases() {
        let src = "use crate::foo::{bar as b, baz as z};\n";
        let result = extract_import_aliases("foo.rs", src);
        // Both aliases must be present; order is traversal order.
        assert!(result.contains(&alias("b", "bar")), "missing b→bar in {result:?}");
        assert!(result.contains(&alias("z", "baz")), "missing z→baz in {result:?}");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn rust_discard_alias_skipped() {
        // `use std::io::Write as _` is a non-binding import suppression.
        let src = "use std::io::Write as _;\n";
        let result = extract_import_aliases("foo.rs", src);
        assert!(result.is_empty(), "expected empty, got {result:?}");
    }

    #[test]
    fn rust_no_alias() {
        let src = "use std::collections::HashMap;\n";
        let result = extract_import_aliases("foo.rs", src);
        assert!(result.is_empty(), "expected empty, got {result:?}");
    }

    #[test]
    fn python_from_import_alias() {
        let src = "from os import path as p\n";
        let result = extract_import_aliases("foo.py", src);
        assert_eq!(result, vec![alias_with_src("p", "path", "os")]);
    }

    #[test]
    fn python_top_level_import_alias() {
        let src = "import json as j\n";
        let result = extract_import_aliases("foo.py", src);
        assert_eq!(result, vec![alias("j", "json")]);
    }

    #[test]
    fn python_relative_import_alias() {
        let src = "from . import foo as bar\n";
        let result = extract_import_aliases("pkg/mod.py", src);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].alias, "bar");
        assert_eq!(result[0].original, "foo");
    }

    #[test]
    fn python_no_alias() {
        let src = "from os import path\nimport sys\n";
        let result = extract_import_aliases("foo.py", src);
        assert!(result.is_empty(), "expected empty, got {result:?}");
    }

    #[test]
    fn typescript_named_import_alias() {
        let src = "import { useState as uS } from 'react';\n";
        let result = extract_import_aliases("foo.ts", src);
        assert_eq!(result, vec![alias_with_src("uS", "useState", "react")]);
    }

    #[test]
    fn typescript_namespace_import() {
        let src = "import * as utils from './utils';\n";
        let result = extract_import_aliases("foo.ts", src);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].alias, "utils");
        // original should be the stem of the import path
        assert_eq!(result[0].original, "utils");
    }

    #[test]
    fn typescript_no_alias() {
        let src = "import { useState } from 'react';\n";
        let result = extract_import_aliases("foo.ts", src);
        assert!(result.is_empty(), "expected empty, got {result:?}");
    }

    #[test]
    fn unknown_extension_returns_empty() {
        let result = extract_import_aliases("file.xyz", "anything");
        assert!(result.is_empty());
    }
}