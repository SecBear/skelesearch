use skelesearch_core::symbols::{extract_symbols, SymbolDef};

#[test]
fn extract_rust_symbols() {
    let source = "pub struct Foo {}\nfn bar() {}\nimpl Foo { fn baz(&self) {} }";
    let symbols = extract_symbols("lib.rs", source).unwrap();
    let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"Foo"), "expected Foo in {names:?}");
    assert!(names.contains(&"bar"), "expected bar in {names:?}");
}

#[test]
fn extract_python_symbols() {
    let source =
        "class MyClass:\n    def method(self):\n        pass\n\ndef free_func():\n    pass";
    let symbols = extract_symbols("app.py", source).unwrap();
    let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"MyClass"), "expected MyClass in {names:?}");
    assert!(names.contains(&"free_func"), "expected free_func in {names:?}");
}

#[test]
fn unknown_extension_returns_empty() {
    let symbols = extract_symbols("readme.md", "# Title").unwrap();
    assert!(symbols.is_empty());
}

#[test]
fn empty_source_returns_empty() {
    let symbols = extract_symbols("lib.rs", "").unwrap();
    assert!(symbols.is_empty());
}

#[tokio::test]
async fn symbols_round_trip_through_backend() -> anyhow::Result<()> {
    use skelesearch_core::{CozoBackend, StorageBackend};
    let dir = tempfile::tempdir()?;
    let backend = CozoBackend::open(dir.path().join("index.db"))?;
    backend.initialize(8).await?;

    backend
        .upsert_symbols(&[SymbolDef {
            file_path: "lib.rs".into(),
            name: "Foo".into(),
            kind: "struct".into(),
            start_line: 1,
            end_line: 1,
        }])
        .await?;

    let found = backend.find_symbols("Foo", None).await?;
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, "struct");

    backend.delete_symbols_for_file("lib.rs").await?;
    let found = backend.find_symbols("Foo", None).await?;
    assert!(found.is_empty());

    Ok(())
}

#[tokio::test]
async fn find_symbols_kind_filter() -> anyhow::Result<()> {
    use skelesearch_core::{CozoBackend, StorageBackend};
    let dir = tempfile::tempdir()?;
    let backend = CozoBackend::open(dir.path().join("index.db"))?;
    backend.initialize(8).await?;

    backend
        .upsert_symbols(&[
            SymbolDef {
                file_path: "lib.rs".into(),
                name: "Foo".into(),
                kind: "struct".into(),
                start_line: 1,
                end_line: 1,
            },
            SymbolDef {
                file_path: "lib.rs".into(),
                name: "Foo".into(),
                kind: "function".into(),
                start_line: 5,
                end_line: 7,
            },
        ])
        .await?;

    // Without kind filter: both matches
    let all = backend.find_symbols("Foo", None).await?;
    assert_eq!(all.len(), 2, "expected 2 matches without kind filter");

    // With kind filter: only struct
    let structs = backend.find_symbols("Foo", Some("struct")).await?;
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].kind, "struct");

    Ok(())
}
