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
// Tier 1: Java
// ---------------------------------------------------------------------------

pub struct JavaConfig;

impl LanguageConfig for JavaConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["java"]
    }

    fn language(&self) -> Language {
        tree_sitter_java::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &[
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        ]
    }

    fn import_query(&self) -> &str {
        "(import_declaration) @path"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: C
// ---------------------------------------------------------------------------

pub struct CConfig;

impl LanguageConfig for CConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["c", "h"]
    }

    fn language(&self) -> Language {
        tree_sitter_c::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &["function_definition", "struct_specifier", "declaration"]
    }

    fn import_query(&self) -> &str {
        "(preproc_include) @path"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: C++
// ---------------------------------------------------------------------------

pub struct CppConfig;

impl LanguageConfig for CppConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["cpp", "cc", "cxx", "hpp"]
    }

    fn language(&self) -> Language {
        tree_sitter_cpp::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &[
            "function_definition",
            "class_specifier",
            "struct_specifier",
        ]
    }

    fn import_query(&self) -> &str {
        "(preproc_include) @path"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: Ruby
// ---------------------------------------------------------------------------

pub struct RubyConfig;

impl LanguageConfig for RubyConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["rb"]
    }

    fn language(&self) -> Language {
        tree_sitter_ruby::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &["method", "singleton_method", "class", "module"]
    }

    fn import_query(&self) -> &str {
        // require/require_relative are method calls in Ruby
        r#"(call
            method: (identifier) @_fn
            arguments: (argument_list (string) @path)
            (#match? @_fn "^require"))"#
    }
}

// ---------------------------------------------------------------------------
// Tier 1: PHP
// The crate exposes LANGUAGE_PHP (full PHP with tags) and LANGUAGE_PHP_ONLY
// (pure PHP without the surrounding HTML context).
// ---------------------------------------------------------------------------

pub struct PhpConfig;

impl LanguageConfig for PhpConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["php"]
    }

    fn language(&self) -> Language {
        tree_sitter_php::LANGUAGE_PHP.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &[
            "function_definition",
            "method_declaration",
            "class_declaration",
            "interface_declaration",
        ]
    }

    fn import_query(&self) -> &str {
        "(namespace_use_declaration) @path"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: C#
// ---------------------------------------------------------------------------

pub struct CSharpConfig;

impl LanguageConfig for CSharpConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["cs"]
    }

    fn language(&self) -> Language {
        tree_sitter_c_sharp::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &[
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "interface_declaration",
        ]
    }

    fn import_query(&self) -> &str {
        "(using_directive) @path"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: Kotlin
// Uses tree-sitter-kotlin-sg (compatible with tree-sitter ≥0.24).
// Note: tree-sitter-kotlin (the original crate) depends on tree-sitter 0.20
// and is ABI-incompatible with tree-sitter 0.26.
// ---------------------------------------------------------------------------

pub struct KotlinConfig;

impl LanguageConfig for KotlinConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["kt", "kts"]
    }

    fn language(&self) -> Language {
        tree_sitter_kotlin_sg::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &[
            "function_declaration",
            "class_declaration",
            "object_declaration",
        ]
    }

    fn import_query(&self) -> &str {
        "(import_header) @path"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: Swift
// ---------------------------------------------------------------------------

pub struct SwiftConfig;

impl LanguageConfig for SwiftConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["swift"]
    }

    fn language(&self) -> Language {
        tree_sitter_swift::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &[
            "function_declaration",
            "class_declaration",
            "struct_declaration",
            "protocol_declaration",
        ]
    }

    fn import_query(&self) -> &str {
        "(import_declaration) @path"
    }
}

// ---------------------------------------------------------------------------
// Tier 1: Scala
// ---------------------------------------------------------------------------

pub struct ScalaConfig;

impl LanguageConfig for ScalaConfig {
    fn file_extensions(&self) -> &[&'static str] {
        &["scala", "sc"]
    }

    fn language(&self) -> Language {
        tree_sitter_scala::LANGUAGE.into()
    }

    fn chunk_node_kinds(&self) -> &[&'static str] {
        &[
            "function_definition",
            "class_definition",
            "object_definition",
            "trait_definition",
        ]
    }

    fn import_query(&self) -> &str {
        "(import_declaration) @path"
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
        "java" => Some(Box::new(JavaConfig)),
        "c" | "h" => Some(Box::new(CConfig)),
        "cpp" | "cc" | "cxx" | "hpp" => Some(Box::new(CppConfig)),
        "rb" => Some(Box::new(RubyConfig)),
        "php" => Some(Box::new(PhpConfig)),
        "cs" => Some(Box::new(CSharpConfig)),
        "kt" | "kts" => Some(Box::new(KotlinConfig)),
        "swift" => Some(Box::new(SwiftConfig)),
        "scala" | "sc" => Some(Box::new(ScalaConfig)),
        _ => None,
    }
}
