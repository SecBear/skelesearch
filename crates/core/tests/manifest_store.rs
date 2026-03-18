use skelesearch_core::ManifestStore;
use std::collections::HashSet;

#[test]
fn manifest_detects_changed_and_deleted_files() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let manifest = ManifestStore::open(temp.path().join("manifest.db"))?;

    manifest.upsert("src/lib.rs", 10, 100, "hash-a")?;
    assert!(manifest.is_unchanged("src/lib.rs", 10, 100, "hash-a")?);
    // Changed mtime.
    assert!(!manifest.is_unchanged("src/lib.rs", 11, 100, "hash-a")?);
    // Changed hash.
    assert!(!manifest.is_unchanged("src/lib.rs", 10, 100, "hash-b")?);
    // Unknown file.
    assert!(!manifest.is_unchanged("src/other.rs", 10, 100, "hash-a")?);

    assert_eq!(manifest.list_paths()?, vec!["src/lib.rs".to_string()]);
    Ok(())
}

#[test]
fn stale_paths_against_reports_removed_paths() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let manifest = ManifestStore::open(temp.path().join("manifest.db"))?;

    manifest.upsert("src/lib.rs", 10, 100, "hash-a")?;

    let mut visited = HashSet::new();
    visited.insert("src/new.rs".into());

    assert_eq!(
        manifest.stale_paths_against(&visited)?,
        vec!["src/lib.rs".to_string()]
    );
    Ok(())
}

#[test]
fn stale_paths_against_treats_rename_as_old_path_becoming_stale() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let manifest = ManifestStore::open(temp.path().join("manifest.db"))?;

    manifest.upsert("src/lib.rs", 10, 100, "hash-a")?;

    let mut visited = HashSet::new();
    visited.insert("src/renamed.rs".into());

    let stale = manifest.stale_paths_against(&visited)?;
    assert!(stale.contains(&"src/lib.rs".to_string()));
    Ok(())
}

#[test]
fn manifest_remove_deletes_entry() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let manifest = ManifestStore::open(temp.path().join("manifest.db"))?;

    manifest.upsert("src/lib.rs", 10, 100, "hash-a")?;
    manifest.remove("src/lib.rs")?;

    assert!(manifest.list_paths()?.is_empty());
    assert!(!manifest.is_unchanged("src/lib.rs", 10, 100, "hash-a")?);
    Ok(())
}


#[test]
fn concurrent_manifest_access_no_busy_errors() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("manifest.db");
    let store1 = ManifestStore::open(&db_path).unwrap();
    let store2 = ManifestStore::open(&db_path).unwrap();
    for i in 0..100 {
        let path = format!("file_{i}.rs");
        store1.upsert(&path, i as i64, 100, "hash_a").unwrap();
        store2.upsert(&path, i as i64, 200, "hash_b").unwrap();
    }
    let paths = store1.list_paths().unwrap();
    assert_eq!(paths.len(), 100);
}

#[test]
fn count_stale_empty_manifest_returns_zero() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    let manifest = ManifestStore::open(dir.path().join("manifest.db")).unwrap();
    assert_eq!(manifest.count_stale(&root).unwrap(), 0);
}

#[test]
fn count_stale_detects_modified_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.rs"), "fn a() {}").unwrap();
    std::fs::write(root.join("b.rs"), "fn b() {}").unwrap();
    let manifest = ManifestStore::open(dir.path().join("manifest.db")).unwrap();
    // Store with mtime=1000, but actual file mtime is current time
    manifest.upsert("a.rs", 1000, 10, "hash_a").unwrap();
    manifest.upsert("b.rs", 1000, 10, "hash_b").unwrap();
    let stale = manifest.count_stale(&root).unwrap();
    assert!(stale >= 1, "expected at least 1 stale file, got {stale}");
}

#[test]
fn count_stale_counts_deleted_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    let manifest = ManifestStore::open(dir.path().join("manifest.db")).unwrap();
    manifest.upsert("deleted.rs", 1000, 10, "hash").unwrap();
    // deleted.rs doesn't exist on disk
    assert_eq!(manifest.count_stale(&root).unwrap(), 1);
}