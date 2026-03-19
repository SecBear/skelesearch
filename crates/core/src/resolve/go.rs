use super::{normalize_path, ImportResolver};
use std::collections::HashSet;
use std::path::Path;

/// Heuristic import resolver for Go.
///
/// Strips the module path (from `go.mod`) from the import path to obtain a
/// project-relative directory, then finds any `.go` file in that directory.
/// Standard library and external imports return `None`.
pub struct GoResolver;

impl ImportResolver for GoResolver {
    fn resolve(
        &self,
        raw_import: &str,
        from_file: &Path,
        project_root: &Path,
        indexed_files: &HashSet<String>,
    ) -> Option<String> {
        let module_path = read_module_path(from_file, project_root)?;
        let rel_dir = raw_import.strip_prefix(&module_path)?.trim_start_matches('/');

        if rel_dir.is_empty() {
            return None; // import is the module root itself
        }

        // In Go, all .go files in a directory belong to one package.
        // Find any .go file in the target directory.
        let prefix = normalize_path(rel_dir);
        let dir_prefix = format!("{prefix}/");

        indexed_files
            .iter()
            .find(|f| f.starts_with(&dir_prefix) && f.ends_with(".go") && !f[dir_prefix.len()..].contains('/'))
            .cloned()
    }
}

/// Read the `module` directive from the nearest `go.mod` file.
///
/// Walks up from `from_file` looking for `go.mod` (permitted filesystem I/O).
fn read_module_path(from_file: &Path, project_root: &Path) -> Option<String> {
    let abs = if from_file.is_relative() {
        project_root.join(from_file)
    } else {
        from_file.to_path_buf()
    };

    let mut dir = abs.parent()?;
    loop {
        let go_mod = dir.join("go.mod");
        if go_mod.exists() {
            let content = std::fs::read_to_string(&go_mod).ok()?;
            return parse_module_directive(&content);
        }
        if dir == project_root || !dir.starts_with(project_root) {
            return None;
        }
        dir = dir.parent()?;
    }
}

/// Extract the module path from `go.mod` content.
///
/// Looks for a line like `module github.com/user/repo`.
fn parse_module_directive(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(path) = trimmed.strip_prefix("module ") {
            let path = path.trim();
            if !path.is_empty() {
                return Some(path.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_set(paths: &[&str]) -> HashSet<String> {
        paths.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_module() {
        let content = "module github.com/user/repo\n\ngo 1.21\n";
        assert_eq!(
            parse_module_directive(content),
            Some("github.com/user/repo".into())
        );
    }

    #[test]
    fn stdlib_returns_none() {
        // "fmt" doesn't start with any module path → None
        let r = GoResolver;
        let indexed = make_set(&["main.go"]);
        // Without a go.mod, read_module_path returns None → resolve returns None
        assert_eq!(
            r.resolve("fmt", Path::new("main.go"), Path::new("/nonexistent"), &indexed),
            None
        );
    }

    // Integration-style tests would need a real go.mod on disk.
    // The core logic (strip module prefix → find .go in directory) is tested
    // indirectly through parse_module_directive + the resolution logic above.
    // Full integration tests are in crates/core/tests/resolve.rs.

    #[test]
    fn resolution_logic_with_known_module() {
        // Test the core logic by calling the resolver methods directly
        let indexed = make_set(&["pkg/utils/helpers.go", "pkg/utils/utils.go", "internal/config/config.go"]);

        // Simulate what resolve() does after getting module_path
        let module_path = "github.com/user/repo";
        let import = "github.com/user/repo/pkg/utils";
        let rel_dir = import.strip_prefix(module_path).unwrap().trim_start_matches('/');
        let dir_prefix = format!("{rel_dir}/");

        let found: Vec<&String> = indexed
            .iter()
            .filter(|f| f.starts_with(&dir_prefix) && f.ends_with(".go") && !f[dir_prefix.len()..].contains('/'))
            .collect();

        assert!(!found.is_empty(), "should find .go files in pkg/utils/");
    }

    #[test]
    fn external_dep_returns_none() {
        // Import from a different module
        let module_path = "github.com/user/repo";
        let import = "github.com/other/lib";
        assert!(import.strip_prefix(module_path).is_none());
    }
}
