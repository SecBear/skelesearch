use super::{normalize_path, ImportResolver};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Heuristic import resolver for Rust source files.
///
/// Resolves `crate::`, `self::`, and `super::` paths to project-relative `.rs`
/// files using Rust's module-to-filesystem conventions.  External crates (no
/// recognised prefix) return `None`.
pub struct RustResolver;

impl ImportResolver for RustResolver {
    fn resolve(
        &self,
        raw_import: &str,
        from_file: &Path,
        project_root: &Path,
        indexed_files: &HashSet<String>,
    ) -> Option<String> {
        let segments: Vec<&str> = raw_import.split("::").collect();
        if segments.is_empty() {
            return None;
        }

        match segments[0] {
            "crate" => {
                let src = find_crate_src_dir(from_file, project_root)?;
                resolve_from_base(&src, &segments[1..], project_root, indexed_files)
            }
            "self" => {
                let base = module_dir(from_file, project_root);
                resolve_from_base(&base, &segments[1..], project_root, indexed_files)
            }
            "super" => {
                let mut base = module_dir(from_file, project_root);
                let mut rest = &segments[1..];
                // First super already walks up once from module_dir
                base = match base.parent() {
                    Some(p) => p.to_path_buf(),
                    None => return None,
                };
                while rest.first() == Some(&"super") {
                    base = match base.parent() {
                        Some(p) => p.to_path_buf(),
                        None => return None,
                    };
                    rest = &rest[1..];
                }
                resolve_from_base(&base, rest, project_root, indexed_files)
            }
            _ => None, // external crate
        }
    }
}

/// Try to resolve `segments` as a module path under `base`.
///
/// Tries, in order:
/// 1. `base/a/b/c.rs`       (all segments are modules)
/// 2. `base/a/b/c/mod.rs`
/// 3. `base/a/b.rs`         (last segment is an item in module b)
/// 4. `base/a/b/mod.rs`
fn resolve_from_base(
    base: &Path,
    segments: &[&str],
    project_root: &Path,
    indexed_files: &HashSet<String>,
) -> Option<String> {
    if segments.is_empty() {
        return None;
    }

    // All segments as modules
    let full: PathBuf = segments.iter().collect();
    let dir = base.join(&full);

    if let Some(r) = check(dir.with_extension("rs"), project_root, indexed_files) {
        return Some(r);
    }
    if let Some(r) = check(dir.join("mod.rs"), project_root, indexed_files) {
        return Some(r);
    }

    // Drop last segment (it might be an item, not a module)
    if segments.len() > 1 {
        let parent: PathBuf = segments[..segments.len() - 1].iter().collect();
        let dir = base.join(&parent);

        if let Some(r) = check(dir.with_extension("rs"), project_root, indexed_files) {
            return Some(r);
        }
        if let Some(r) = check(dir.join("mod.rs"), project_root, indexed_files) {
            return Some(r);
        }
    }

    None
}

fn check(path: PathBuf, root: &Path, indexed: &HashSet<String>) -> Option<String> {
    // Path may be absolute (root.join(base).join(segments)) or relative — normalize.
    let rel = if path.is_absolute() {
        path.strip_prefix(root).ok()?.to_path_buf()
    } else {
        path
    };
    let s = normalize_path(&rel.to_string_lossy());
    if indexed.contains(&s) { Some(s) } else { None }
}

/// Find the `src/` directory of the crate containing `from_file`.
///
/// Walks up from the file looking for `Cargo.toml` (the one permitted filesystem
/// check), then returns `project_root / crate_dir / src`.
fn find_crate_src_dir(from_file: &Path, project_root: &Path) -> Option<PathBuf> {
    let abs = if from_file.is_relative() {
        project_root.join(from_file)
    } else {
        from_file.to_path_buf()
    };

    let mut dir = abs.parent()?;
    loop {
        if dir.join("Cargo.toml").exists() {
            return Some(dir.join("src"));
        }
        if dir == project_root || !dir.starts_with(project_root) {
            // Fallback: assume src/ at project root.
            return Some(project_root.join("src"));
        }
        dir = dir.parent()?;
    }
}

/// The directory representing the module that `from_file` belongs to.
///
/// - `src/foo/bar.rs` → `src/foo`
/// - `src/foo/mod.rs` → `src/foo`
/// - `src/lib.rs`     → `src`
fn module_dir(from_file: &Path, project_root: &Path) -> PathBuf {
    let abs = if from_file.is_relative() {
        project_root.join(from_file)
    } else {
        from_file.to_path_buf()
    };
    abs.parent().unwrap_or(&abs).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_set(paths: &[&str]) -> HashSet<String> {
        paths.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn crate_schema_resolves() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/schema.rs", "src/lib.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("crate::schema", Path::new("src/lib.rs"), root, &indexed),
            Some("src/schema.rs".into())
        );
    }

    #[test]
    fn crate_nested_module() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/chunker/languages.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("crate::chunker::languages", Path::new("src/lib.rs"), root, &indexed),
            Some("src/chunker/languages.rs".into())
        );
    }

    #[test]
    fn crate_item_in_module() {
        // foo::bar where bar is an item in foo.rs (no src/foo/bar.rs exists)
        let root = Path::new("/project");
        let indexed = make_set(&["src/foo.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("crate::foo::bar", Path::new("src/lib.rs"), root, &indexed),
            Some("src/foo.rs".into())
        );
    }

    #[test]
    fn crate_mod_rs_variant() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/foo/bar/mod.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("crate::foo::bar", Path::new("src/lib.rs"), root, &indexed),
            Some("src/foo/bar/mod.rs".into())
        );
    }

    #[test]
    fn self_utils() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/resolve/utils.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("self::utils", Path::new("src/resolve/mod.rs"), root, &indexed),
            Some("src/resolve/utils.rs".into())
        );
    }

    #[test]
    fn super_schema() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/schema.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("super::schema", Path::new("src/resolve/rust.rs"), root, &indexed),
            Some("src/schema.rs".into())
        );
    }

    #[test]
    fn external_crate_returns_none() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/lib.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("anyhow::Result", Path::new("src/lib.rs"), root, &indexed),
            None
        );
    }
}
