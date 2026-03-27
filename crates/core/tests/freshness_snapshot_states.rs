use skelesearch_core::{FreshnessSnapshot, FreshnessState, ManifestStore};

#[test]
fn freshness_snapshot_states_empty_manifest_is_fresh_when_check_runs() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root)?;

    let manifest = ManifestStore::open(dir.path().join("manifest.db"))?;
    let snapshot = FreshnessSnapshot::from_manifest(&manifest, &root);

    assert_eq!(snapshot.state, FreshnessState::Fresh);
    assert_eq!(snapshot.estimated_stale, 0);
    assert!(snapshot.freshness_checked_at.is_some());
    assert!(snapshot.freshness_error.is_none());
    Ok(())
}

#[test]
fn freshness_snapshot_states_fresh_manifest_entry_is_fresh() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root)?;
    let file = root.join("src/lib.rs");
    std::fs::create_dir_all(file.parent().expect("parent exists"))?;
    std::fs::write(&file, "fn fresh() {}")?;

    let mtime = std::fs::metadata(&file)?
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs() as i64;

    let manifest = ManifestStore::open(dir.path().join("manifest.db"))?;
    manifest.upsert("src/lib.rs", mtime, 13, "hash")?;

    let snapshot = FreshnessSnapshot::from_manifest(&manifest, &root);
    assert_eq!(snapshot.state, FreshnessState::Fresh);
    assert_eq!(snapshot.estimated_stale, 0);
    assert!(snapshot.freshness_checked_at.is_some());
    assert!(snapshot.freshness_error.is_none());
    Ok(())
}

#[test]
fn freshness_snapshot_states_detects_stale_entries() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root)?;
    let file = root.join("src/lib.rs");
    std::fs::create_dir_all(file.parent().expect("parent exists"))?;
    std::fs::write(&file, "fn stale() {}")?;

    let manifest = ManifestStore::open(dir.path().join("manifest.db"))?;
    manifest.upsert("src/lib.rs", 0, 13, "hash")?;

    let snapshot = FreshnessSnapshot::from_manifest(&manifest, &root);
    assert_eq!(snapshot.state, FreshnessState::Stale);
    assert!(snapshot.estimated_stale >= 1);
    assert!(snapshot.freshness_checked_at.is_some());
    assert!(snapshot.freshness_error.is_none());
    Ok(())
}

#[test]
fn freshness_snapshot_states_check_failed_is_unknown() {
    let snapshot = FreshnessSnapshot::from_stale_count_result(Err(anyhow::anyhow!("count failed")));

    assert_eq!(snapshot.state, FreshnessState::Unknown);
    assert_eq!(snapshot.estimated_stale, 0);
    assert!(snapshot.freshness_checked_at.is_none());
    assert!(snapshot
        .freshness_error
        .as_deref()
        .unwrap_or_default()
        .contains("count failed"));
}

#[test]
fn freshness_snapshot_states_can_be_marked_refreshing() {
    let snapshot = FreshnessSnapshot::from_stale_count_result(Ok(0)).with_refreshing(true);

    assert_eq!(snapshot.state, FreshnessState::Refreshing);
    assert_eq!(snapshot.estimated_stale, 0);
    assert!(snapshot.freshness_checked_at.is_some());
    assert!(snapshot.freshness_error.is_none());
}
