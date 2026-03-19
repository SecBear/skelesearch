use super::ImportResolver;
use std::collections::HashSet;
use std::path::Path;

/// Heuristic import resolver for Python.
///
/// Maps dotted module names (`foo.bar.baz`) to file paths by directory
/// convention.  Returns `None` for stdlib and third-party imports (anything
/// not found in the project tree).
pub struct PythonResolver;

impl ImportResolver for PythonResolver {
    fn resolve(
        &self,
        raw_import: &str,
        _from_file: &Path,
        _project_root: &Path,
        indexed_files: &HashSet<String>,
    ) -> Option<String> {
        // Leading dots indicate relative imports — tree-sitter captures them as
        // part of the dotted_name.  Relative resolution is deferred; the dots
        // cause empty segments which we bail on.
        let segments: Vec<&str> = raw_import.split('.').collect();
        if segments.is_empty() || segments.iter().any(|s| s.is_empty()) {
            return None;
        }

        // Try from project root, then with src/ prefix (src-layout projects).
        for prefix in &["", "src/"] {
            if let Some(r) = try_resolve(prefix, &segments, indexed_files) {
                return Some(r);
            }
        }

        None
    }
}

/// Try all-segments-as-modules, then drop-last (last segment is an attribute).
fn try_resolve(
    prefix: &str,
    segments: &[&str],
    indexed_files: &HashSet<String>,
) -> Option<String> {
    // All segments as modules: foo/bar/baz.py or foo/bar/baz/__init__.py
    let full_path = format!("{}{}", prefix, segments.join("/"));

    let py = format!("{full_path}.py");
    if indexed_files.contains(&py) {
        return Some(py);
    }
    let init = format!("{full_path}/__init__.py");
    if indexed_files.contains(&init) {
        return Some(init);
    }

    // Drop last segment (it might be a symbol/attribute in the parent module)
    if segments.len() > 1 {
        let parent_path = format!("{}{}", prefix, segments[..segments.len() - 1].join("/"));

        let py = format!("{parent_path}.py");
        if indexed_files.contains(&py) {
            return Some(py);
        }
        let init = format!("{parent_path}/__init__.py");
        if indexed_files.contains(&init) {
            return Some(init);
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

    fn resolve(import: &str, indexed: &HashSet<String>) -> Option<String> {
        let r = PythonResolver;
        r.resolve(import, Path::new("main.py"), Path::new("/project"), indexed)
    }

    #[test]
    fn stdlib_returns_none() {
        assert_eq!(resolve("os.path", &make_set(&[])), None);
    }

    #[test]
    fn resolves_module_file() {
        let indexed = make_set(&["myapp/models/user.py"]);
        assert_eq!(resolve("myapp.models.user", &indexed), Some("myapp/models/user.py".into()));
    }

    #[test]
    fn resolves_package_init() {
        let indexed = make_set(&["myapp/models/__init__.py"]);
        assert_eq!(resolve("myapp.models", &indexed), Some("myapp/models/__init__.py".into()));
    }

    #[test]
    fn resolves_src_layout() {
        let indexed = make_set(&["src/myapp/utils.py"]);
        assert_eq!(resolve("myapp.utils", &indexed), Some("src/myapp/utils.py".into()));
    }

    #[test]
    fn attribute_in_module() {
        // myapp.models.User where User is a class in models.py
        let indexed = make_set(&["myapp/models.py"]);
        assert_eq!(resolve("myapp.models.User", &indexed), Some("myapp/models.py".into()));
    }

    #[test]
    fn prefers_exact_module_over_parent() {
        // If both myapp/models/user.py and myapp/models.py exist, prefer the exact match
        let indexed = make_set(&["myapp/models/user.py", "myapp/models.py"]);
        assert_eq!(resolve("myapp.models.user", &indexed), Some("myapp/models/user.py".into()));
    }

    #[test]
    fn relative_import_dots_return_none() {
        // Leading dots produce empty segments
        let indexed = make_set(&["myapp/utils.py"]);
        assert_eq!(resolve(".utils", &indexed), None);
    }
}
