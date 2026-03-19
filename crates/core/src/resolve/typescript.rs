use super::{normalize_path, ImportResolver};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Heuristic import resolver for TypeScript and JavaScript.
///
/// Handles relative imports (`./foo`, `../bar`) with extension probing and
/// directory index resolution.  Bare specifiers (`react`) and aliases (`@/...`)
/// return `None`.
pub struct TypeScriptResolver;

/// Extensions to probe, in priority order.  JS→TS rewriting comes first so
/// that `import './foo.js'` prefers the `.ts` variant when both exist.
const TS_EXTENSIONS: &[&str] = &[".ts", ".tsx", ".js", ".jsx"];
const INDEX_FILES: &[&str] = &["index.ts", "index.tsx", "index.js"];

impl ImportResolver for TypeScriptResolver {
    fn resolve(
        &self,
        raw_import: &str,
        from_file: &Path,
        project_root: &Path,
        indexed_files: &HashSet<String>,
    ) -> Option<String> {
        // Only resolve relative imports.
        if !raw_import.starts_with("./") && !raw_import.starts_with("../") {
            return None;
        }

        let from_dir = from_file.parent().unwrap_or(Path::new(""));
        let base = resolve_relative(from_dir, raw_import, project_root);

        // 1. Exact match (rare — import already has extension)
        if let Some(r) = probe(&base, project_root, indexed_files) {
            return Some(r);
        }

        // 2. JS→TS rewriting: if import ends with .js, try .ts/.tsx first
        let base_str = base.to_string_lossy();
        if base_str.ends_with(".js") {
            let stem = &base_str[..base_str.len() - 3];
            for ext in &[".ts", ".tsx"] {
                let candidate = PathBuf::from(format!("{stem}{ext}"));
                if let Some(r) = probe(&candidate, project_root, indexed_files) {
                    return Some(r);
                }
            }
            // Also try the original .js
            if let Some(r) = probe(&base, project_root, indexed_files) {
                return Some(r);
            }
        }

        // 3. Extension probing
        for ext in TS_EXTENSIONS {
            let candidate = PathBuf::from(format!("{}{ext}", base.display()));
            if let Some(r) = probe(&candidate, project_root, indexed_files) {
                return Some(r);
            }
        }

        // 4. Directory index
        for index in INDEX_FILES {
            let candidate = base.join(index);
            if let Some(r) = probe(&candidate, project_root, indexed_files) {
                return Some(r);
            }
        }

        None
    }
}

/// Resolve `.` and `..` components without filesystem I/O.
fn resolve_relative(from_dir: &Path, import: &str, project_root: &Path) -> PathBuf {
    let abs_dir = if from_dir.is_relative() {
        project_root.join(from_dir)
    } else {
        from_dir.to_path_buf()
    };

    let mut result = abs_dir;
    for component in Path::new(import).components() {
        match component {
            std::path::Component::CurDir => {} // .
            std::path::Component::ParentDir => {
                result.pop();
            }
            std::path::Component::Normal(seg) => result.push(seg),
            _ => {}
        }
    }
    result
}

fn probe(path: &Path, root: &Path, indexed: &HashSet<String>) -> Option<String> {
    let rel = if path.is_absolute() {
        path.strip_prefix(root).ok()?.to_path_buf()
    } else {
        path.to_path_buf()
    };
    let s = normalize_path(&rel.to_string_lossy());
    if indexed.contains(&s) { Some(s) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_set(paths: &[&str]) -> HashSet<String> {
        paths.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn relative_ts_file() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/components/utils.ts"]);
        let r = TypeScriptResolver;
        assert_eq!(
            r.resolve("./utils", Path::new("src/components/Button.tsx"), root, &indexed),
            Some("src/components/utils.ts".into())
        );
    }

    #[test]
    fn parent_relative() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/shared/types.ts"]);
        let r = TypeScriptResolver;
        assert_eq!(
            r.resolve("../shared/types", Path::new("src/components/Button.tsx"), root, &indexed),
            Some("src/shared/types.ts".into())
        );
    }

    #[test]
    fn directory_index() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/components/utils/index.ts"]);
        let r = TypeScriptResolver;
        assert_eq!(
            r.resolve("./utils", Path::new("src/components/Button.tsx"), root, &indexed),
            Some("src/components/utils/index.ts".into())
        );
    }

    #[test]
    fn js_to_ts_rewriting() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/components/foo.ts"]);
        let r = TypeScriptResolver;
        assert_eq!(
            r.resolve("./foo.js", Path::new("src/components/Bar.ts"), root, &indexed),
            Some("src/components/foo.ts".into())
        );
    }

    #[test]
    fn bare_specifier_returns_none() {
        let root = Path::new("/project");
        let indexed = make_set(&["node_modules/react/index.js"]);
        let r = TypeScriptResolver;
        assert_eq!(
            r.resolve("react", Path::new("src/App.tsx"), root, &indexed),
            None
        );
    }

    #[test]
    fn alias_returns_none() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/utils.ts"]);
        let r = TypeScriptResolver;
        assert_eq!(
            r.resolve("@/utils", Path::new("src/App.tsx"), root, &indexed),
            None
        );
    }

    #[test]
    fn exact_extension_match() {
        let root = Path::new("/project");
        let indexed = make_set(&["src/data.json"]);
        let r = TypeScriptResolver;
        assert_eq!(
            r.resolve("./data.json", Path::new("src/App.ts"), root, &indexed),
            Some("src/data.json".into())
        );
    }
}
