use tree_sitter::Language;

/// Per-language configuration for AST-aware chunking and import extraction.
///
/// Tier 1 languages have explicit implementations.  Unknown extensions fall
/// back to the Tier 2 sliding-window splitter in `mod.rs`.
pub trait LanguageConfig: Send + Sync {
    /// File extensions handled by this config (without the leading dot).
    fn file_extensions(&self) -> &[&'static str];

    /// The tree-sitter `Language` used for parsing and queries.
    fn language(&self) -> Language;

    /// AST node kinds that form chunk boundaries (e.g. `"function_item"`).
    fn chunk_node_kinds(&self) -> &[&'static str];

    /// An S-expression tree-sitter query that captures import target nodes
    /// as `@path`.  The captured text is the raw import path string (possibly
    /// including surrounding quotes; callers must strip them).
    fn import_query(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Tier 1: Rust
// ---------------------------------------------------------------------------

pub struct RustConfig;

impl LanguageConfig for RustConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["rs"]
    }

    fn language(&self) -> Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &["function_item", "impl_item", "struct_item", "trait_item", "enum_item"]
    }

    fn import_query(&self) -> &str {
        "(use_declaration) @path"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: Nix
// ---------------------------------------------------------------------------

pub struct NixConfig;

impl LanguageConfig for NixConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["nix"]
    }

    fn language(&self) -> Language {
        tree_sitter_nix::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &["function_expression", "let_expression", "attrset_expression"]
    }

    fn import_query(&self) -> &str {
        // Nix `import` is a built-in function call: `import ./foo/bar.nix`
        "(apply_expression
            function: (variable_expression) @_fn
            argument: [(path_expression) (string_expression)] @path
            (#eq? @_fn \"import\"))"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: Python
// ---------------------------------------------------------------------------

pub struct PythonConfig;

impl LanguageConfig for PythonConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["py"]
    }

    fn language(&self) -> Language {
        tree_sitter_python::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &["function_definition", "class_definition"]
    }

    fn import_query(&self) -> &str {
        r#"[
            (import_statement name: (dotted_name) @path)
            (import_from_statement module_name: (dotted_name) @path)
        ]"#
    }
}

// ---------------------------------------------------------------------------
// Tier 1: TypeScript
// ---------------------------------------------------------------------------

pub struct TypeScriptConfig;

impl LanguageConfig for TypeScriptConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["ts", "tsx"]
    }

    fn language(&self) -> Language {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &["function_declaration", "class_declaration", "method_definition"]
    }

    fn import_query(&self) -> &str {
        "(import_statement source: (string) @path)"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: JavaScript
// ---------------------------------------------------------------------------

pub struct JavaScriptConfig;

impl LanguageConfig for JavaScriptConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["js", "jsx"]
    }

    fn language(&self) -> Language {
        tree_sitter_javascript::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &["function_declaration", "class_declaration"]
    }

    fn import_query(&self) -> &str {
        "(import_statement source: (string) @path)"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: Go
// ---------------------------------------------------------------------------

pub struct GoConfig;

impl LanguageConfig for GoConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["go"]
    }

    fn language(&self) -> Language {
        tree_sitter_go::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &["function_declaration", "method_declaration", "type_declaration"]
    }

    fn import_query(&self) -> &str {
        r#"(import_spec path: (interpreted_string_literal) @path)"#
    }
}

// ---------------------------------------------------------------------------
// Registry: extension → LanguageConfig
// ---------------------------------------------------------------------------

/// Return the `LanguageConfig` for `extension` (without leading dot), or
/// `None` if the extension is not in the Tier 1 registry.
pub fn config_for_extension(extension: &str) -> Option<Box<dyn LanguageConfig>> {
    match extension {
        "rs" => Some(Box::new(RustConfig)),
        "nix" => Some(Box::new(NixConfig)),
        "py" => Some(Box::new(PythonConfig)),
        "ts" | "tsx" => Some(Box::new(TypeScriptConfig)),
        "js" | "jsx" => Some(Box::new(JavaScriptConfig)),
        "go" => Some(Box::new(GoConfig)),
        _ => None,
    }
}
