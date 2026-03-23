use super::{normalize_path, ImportResolver};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Heuristic import resolver for Rust source files.
///
/// Resolves `crate::`, `self::`, and `super::` paths to project-relative `.rs`
/// files using Rust's module-to-filesystem conventions.  For paths with an
/// unrecognised first segment (external crates), workspace-aware resolution is
/// attempted: if the segment matches a sibling workspace member's package name
/// (with `'-'` mapped to `'_'`), the remaining segments are resolved against
/// that member's `src/` directory.  Truly external crates (not a workspace
/// member) return `None`.
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
            first => resolve_workspace_member(
                first,
                &segments[1..],
                from_file,
                project_root,
                indexed_files,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Workspace-aware external crate resolution
// ---------------------------------------------------------------------------

/// Try to resolve `first_segment::remaining` as a sibling workspace member.
///
/// Returns `None` when:
/// - no `[workspace]` Cargo.toml is reachable from `from_file`, or
/// - no workspace member's package name maps to `first_segment`, or
/// - the remaining path cannot be resolved against the member's `src/`.
///
/// **Assumption:** `project_root` is an ancestor of (or equal to) the
/// workspace root.  If `project_root` is set to a sub-crate directory, paths
/// inside sibling members cannot be stripped to project-relative form and will
/// resolve to `None`.
fn resolve_workspace_member(
    first_segment: &str,
    remaining: &[&str],
    from_file: &Path,
    project_root: &Path,
    indexed_files: &HashSet<String>,
) -> Option<String> {
    let workspace_root = find_workspace_root(from_file, project_root)?;
    let toml_content = std::fs::read_to_string(workspace_root.join("Cargo.toml")).ok()?;
    let members = parse_workspace_members(&toml_content)?;
    let member_dir = find_member_by_name(&workspace_root, &members, first_segment)?;
    // Default to `<member>/src`; handles the vast majority of crates.
    // Custom `[lib] path` overrides are not supported here.
    let src_dir = member_dir.join("src");
    resolve_from_base(&src_dir, remaining, project_root, indexed_files)
}

/// Walk up from `from_file` toward `project_root` looking for the first
/// `Cargo.toml` that contains a `[workspace]` section.
///
/// Returns the directory containing that `Cargo.toml`, or `None` if none is
/// found before going above `project_root`.
fn find_workspace_root(from_file: &Path, project_root: &Path) -> Option<PathBuf> {
    let abs = if from_file.is_relative() {
        project_root.join(from_file)
    } else {
        from_file.to_path_buf()
    };

    let mut dir = abs.parent()?;
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return Some(dir.to_path_buf());
                }
            }
        }
        // Stop after we have checked project_root itself.
        if dir == project_root {
            return None;
        }
        if !dir.starts_with(project_root) {
            return None;
        }
        dir = dir.parent()?;
    }
}

/// Parse the `members` array from a workspace `Cargo.toml` string.
///
/// Performs minimal line-oriented parsing — sufficient for the patterns
/// produced by `cargo new` and `cargo workspaces`:
/// - `members = ["path1", "path2"]`  (inline)
/// - multi-line array with one entry per line
///
/// Returns `None` only if no `[workspace]` section or `members` key exists.
fn parse_workspace_members(toml_content: &str) -> Option<Vec<String>> {
    let ws_start = toml_content.find("[workspace]")?;
    let after_ws = &toml_content[ws_start + "[workspace]".len()..];

    // Bound search to this section; stop at the next bare `[section]` header.
    let section_end = find_section_end(after_ws);
    let ws_body = &after_ws[..section_end];

    // Find `members = [...]`.
    let m_pos = ws_body.find("members")?;
    let after_m = &ws_body[m_pos..];
    let open = after_m.find('[')?;
    let after_open = &after_m[open + 1..];
    let close = after_open.find(']')?;
    let array_str = &after_open[..close];

    // Extract all double-quoted strings from the array literal.
    let mut members = Vec::new();
    let mut s = array_str;
    while let Some(q1) = s.find('"') {
        s = &s[q1 + 1..];
        if let Some(q2) = s.find('"') {
            members.push(s[..q2].to_string());
            s = &s[q2 + 1..];
        } else {
            break;
        }
    }

    // Return None rather than an empty Vec — callers treat empty as "no members"
    // and would loop over nothing, but an explicit None is cleaner.
    Some(members)
}

/// Return the byte offset in `s` where the next TOML section header begins,
/// or `s.len()` if there is none.
///
/// A section header is a line whose first non-whitespace character is `[`
/// (but not `[[`, which is a TOML array-of-tables header that may legitimately
/// appear inside a section we are scanning).
fn find_section_end(s: &str) -> usize {
    let mut offset = match s.find('\n') {
        Some(nl) => nl + 1,
        None => return s.len(),
    };


    while offset < s.len() {
        let rest = &s[offset..];
        let line_end = rest.find('\n').map(|i| i + offset).unwrap_or(s.len());
        let line = &s[offset..line_end];
        let trimmed = line.trim();
        if trimmed.starts_with('[') && !trimmed.starts_with("[[") {
            return offset;
        }
        offset = line_end + 1;
    }
    s.len()
}

/// Iterate over workspace member patterns and return the directory of the
/// first member whose package name maps to `import_name`.
///
/// Rust converts hyphens to underscores in import paths, so `skg-core`
/// is imported as `skg_core`.  This function applies that mapping when
/// comparing package names.
///
/// Glob patterns of the form `prefix/*` are expanded by reading the
/// directory; only one level of wildcard is supported.
fn find_member_by_name(
    workspace_root: &Path,
    members: &[String],
    import_name: &str,
) -> Option<PathBuf> {
    for pattern in members {
        if pattern.ends_with("/*") {
            // Simple single-level glob: enumerate all subdirectories.
            let base = workspace_root.join(&pattern[..pattern.len() - 2]);
            if let Ok(entries) = std::fs::read_dir(&base) {
                for entry in entries.flatten() {
                    let member_dir = entry.path();
                    if member_dir.is_dir() {
                        if let Some(pkg_name) = read_package_name(&member_dir) {
                            if pkg_name.replace('-', "_") == import_name {
                                return Some(member_dir);
                            }
                        }
                    }
                }
            }
        } else {
            let member_dir = workspace_root.join(pattern);
            if let Some(pkg_name) = read_package_name(&member_dir) {
                if pkg_name.replace('-', "_") == import_name {
                    return Some(member_dir);
                }
            }
        }
    }
    None
}

/// Read `member_dir/Cargo.toml` and return the value of `[package] name`.
fn read_package_name(member_dir: &Path) -> Option<String> {
    let content = std::fs::read_to_string(member_dir.join("Cargo.toml")).ok()?;
    parse_package_name(&content)
}

/// Parse the `[package] name` value from a Cargo.toml string.
fn parse_package_name(toml_content: &str) -> Option<String> {
    let pkg_pos = toml_content.find("[package]")?;
    let after_pkg = &toml_content[pkg_pos + "[package]".len()..];
    let section_end = find_section_end(after_pkg);
    let pkg_body = &after_pkg[..section_end];

    for line in pkg_body.lines() {
        let trimmed = line.trim();
        // Match `name = "..."` — guard against matching `name-something`.
        if trimmed.starts_with("name") {
            let rest = trimmed[4..].trim_start();
            if rest.starts_with('=') {
                let after_eq = rest[1..].trim_start();
                if let Some(inner) = after_eq.strip_prefix('"') {
                    if let Some(end) = inner.find('"') {
                        return Some(inner[..end].to_string());
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// resolve_from_base
// ---------------------------------------------------------------------------

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
    use std::fs;

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
        // /project doesn't exist on the filesystem, so find_workspace_root
        // returns None immediately and we fall back to None.
        let root = Path::new("/project");
        let indexed = make_set(&["src/lib.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("anyhow::Result", Path::new("src/lib.rs"), root, &indexed),
            None
        );
    }

    // -----------------------------------------------------------------------
    // Workspace-aware resolution tests
    // -----------------------------------------------------------------------

    /// Build a minimal workspace on disk and return the root path.
    ///
    /// Layout created:
    /// ```
    /// <root>/
    ///   Cargo.toml          # workspace, members = [extra_members..., literal_members...]
    ///   <member_path>/
    ///     Cargo.toml        # [package] name = "<pkg_name>"
    ///     src/
    ///       (empty — we only need the directory to exist for resolution)
    /// ```
    fn create_workspace(
        test_name: &str,
        literal_members: &[(&str, &str)], // (dir_path, package_name)
        glob_prefix: Option<&str>,        // e.g. Some("crates") → member = "crates/*"
        glob_members: &[(&str, &str)],    // subdirs under glob_prefix: (subdir, pkg_name)
    ) -> PathBuf {
        let root = std::env::temp_dir().join(format!("skg_rr_{}", test_name));
        // Clean up any previous run to ensure a fresh state.
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        // Build workspace Cargo.toml members list.
        let mut member_entries: Vec<String> = Vec::new();
        for (dir, _) in literal_members {
            member_entries.push(format!("\"{}\"", dir));
        }
        if let Some(prefix) = glob_prefix {
            member_entries.push(format!("\"{}/*\"", prefix));
            for (subdir, pkg_name) in glob_members {
                let member_dir = root.join(prefix).join(subdir);
                fs::create_dir_all(member_dir.join("src")).unwrap();
                fs::write(
                    member_dir.join("Cargo.toml"),
                    format!("[package]\nname = \"{}\"\nversion = \"0.1.0\"\n", pkg_name),
                )
                .unwrap();
            }
        }
        let ws_toml = format!(
            "[workspace]\nmembers = [{}]\n",
            member_entries.join(", ")
        );
        fs::write(root.join("Cargo.toml"), &ws_toml).unwrap();

        // Create literal members.
        for (dir, pkg_name) in literal_members {
            let member_dir = root.join(dir);
            fs::create_dir_all(member_dir.join("src")).unwrap();
            fs::write(
                member_dir.join("Cargo.toml"),
                format!("[package]\nname = \"{}\"\nversion = \"0.1.0\"\n", pkg_name),
            )
            .unwrap();
        }

        root
    }

    #[test]
    fn workspace_member_resolves() {
        // layer0::dispatch::Dispatcher → layer0/src/dispatch.rs
        let root = create_workspace("ws_member", &[("layer0", "layer0")], None, &[]);
        let indexed = make_set(&["layer0/src/dispatch.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve(
                "layer0::dispatch::Dispatcher",
                Path::new("src/lib.rs"),
                &root,
                &indexed,
            ),
            Some("layer0/src/dispatch.rs".into())
        );
    }

    #[test]
    fn workspace_member_with_hyphen() {
        // Crate `skg-turn` (hyphen) imported as `skg_turn` (underscore).
        // skg_turn::provider → turn/skg-turn/src/provider.rs
        let root = create_workspace(
            "ws_hyphen",
            &[("turn/skg-turn", "skg-turn")],
            None,
            &[],
        );
        let indexed = make_set(&["turn/skg-turn/src/provider.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve(
                "skg_turn::provider",
                Path::new("src/lib.rs"),
                &root,
                &indexed,
            ),
            Some("turn/skg-turn/src/provider.rs".into())
        );
    }

    #[test]
    fn workspace_glob_member() {
        // members = ["crates/*"], crate `skelesearch-core` → skelesearch_core::schema
        let root = create_workspace(
            "ws_glob",
            &[],
            Some("crates"),
            &[("core", "skelesearch-core")],
        );
        let indexed = make_set(&["crates/core/src/schema.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve(
                "skelesearch_core::schema",
                Path::new("src/lib.rs"),
                &root,
                &indexed,
            ),
            Some("crates/core/src/schema.rs".into())
        );
    }

    #[test]
    fn unknown_external_crate() {
        // tokio is not a workspace member → None.
        let root = create_workspace("ws_unknown", &[("layer0", "layer0")], None, &[]);
        let indexed = make_set(&["src/lib.rs"]);
        let r = RustResolver;
        assert_eq!(
            r.resolve("tokio::sync", Path::new("src/lib.rs"), &root, &indexed),
            None
        );
    }
}
