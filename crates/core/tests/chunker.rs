use skelesearch_core::{normalize_for_fts, Chunker};

#[test]
fn rust_chunker_preserves_function_boundaries() -> anyhow::Result<()> {
    let source = r#"
fn alpha() { println!("a"); }
fn beta() { println!("b"); }
"#;
    let chunker = Chunker::default();
    let chunks = chunker.chunk_file("src/lib.rs", source)?;

    assert!(chunks.iter().any(|c| c.content.contains("fn alpha")));
    assert!(chunks.iter().any(|c| c.content.contains("fn beta")));
    Ok(())
}

#[test]
fn normalization_splits_mixed_identifiers() {
    assert_eq!(normalize_for_fts("parseHTTPResponse_json"), "parse http response json");
}

#[test]
fn typescript_import_query_extracts_edges() -> anyhow::Result<()> {
    let source = "import { search } from './search';\nexport const x = search();";
    let edges = Chunker::default().extract_edges("src/app.ts", source)?;
    assert!(!edges.is_empty(), "expected at least one edge");
    assert_eq!(edges[0].to_file, "./search");
    Ok(())
}

#[test]
fn nix_python_javascript_and_go_configs_produce_chunks() -> anyhow::Result<()> {
    assert!(
        !Chunker::default()
            .chunk_file("flake.nix", "let x = 1; in x")?
            .is_empty()
    );
    assert!(
        !Chunker::default()
            .chunk_file("app.py", "def run():\n    return 1\n")?
            .is_empty()
    );
    assert!(
        !Chunker::default()
            .chunk_file("app.js", "function run() { return 1; }")?
            .is_empty()
    );
    assert!(
        !Chunker::default()
            .chunk_file("main.go", "package main\nfunc run() {}")?
            .is_empty()
    );
    Ok(())
}

#[test]
fn unknown_extension_uses_sliding_window_fallback() -> anyhow::Result<()> {
    let source = "word ".repeat(400);
    let chunks = Chunker::default().chunk_file("notes.txt", &source)?;
    assert!(chunks.len() > 1, "expected multiple chunks for long text, got {}", chunks.len());
    Ok(())
}

#[test]
fn typescript_chunker_produces_chunks() -> anyhow::Result<()> {
    let source = r#"
function hello() {
    return "hello";
}
function world() {
    return "world";
}
"#;
    let chunks = Chunker::default().chunk_file("src/app.ts", source)?;
    assert!(!chunks.is_empty());
    Ok(())
}

#[test]
fn normalize_fts_examples() {
    assert_eq!(normalize_for_fts("MyStruct"), "my struct");
    assert_eq!(normalize_for_fts("foo_bar_baz"), "foo bar baz");
    assert_eq!(normalize_for_fts("getHTTPSUrl"), "get https url");
}
