//! Import path resolution: raw tree-sitter capture text → canonical project-relative path.
//!
//! The flow is:
//! 1. `extract_import_path` strips language-specific syntax from the raw capture to obtain the
//!    bare module/path token (e.g. `"crate::foo::bar"`, `"./utils"`, `"os.path"`).
//! 2. A language-specific [`ImportResolver`] maps that token to a project-relative file path by
//!    checking membership in `indexed_files` — no filesystem I/O is performed.
//!
//! All four resolver implementations live in sub-modules that are stubs today.  The dispatch
//! entry point [`resolver_for_extension`] selects the right one from a file extension.

pub mod go;
pub mod python;
pub mod rust;
pub mod typescript;

use std::collections::HashSet;
use std::path::Path;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Resolves a bare import token to a project-relative file path.
///
/// Implementors **must not** perform filesystem I/O; all path membership queries go through
/// `indexed_files`, which contains every path the indexer knows about.
pub trait ImportResolver: Send + Sync {
    /// Attempt to resolve `raw_import` (already stripped of language syntax by
    /// [`extract_import_path`]) to a project-relative path found in `indexed_files`.
    ///
    /// Returns `None` when resolution is impossible or ambiguous (e.g. external packages).
    fn resolve(
        &self,
        raw_import: &str,
        from_file: &Path,
        project_root: &Path,
        indexed_files: &HashSet<String>,
    ) -> Option<String>;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Return the [`ImportResolver`] for the given file extension, or `None` for unknown extensions.
///
/// Recognised extensions: `rs`, `ts`, `tsx`, `js`, `jsx`, `mjs`, `cjs`, `py`, `go`.
pub fn resolver_for_extension(ext: &str) -> Option<Box<dyn ImportResolver>> {
    match ext {
        "rs" => Some(Box::new(rust::RustResolver)),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => {
            Some(Box::new(typescript::TypeScriptResolver))
        }
        "py" | "pyi" => Some(Box::new(python::PythonResolver)),
        "go" => Some(Box::new(go::GoResolver)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// extract_import_path
// ---------------------------------------------------------------------------

/// Strip language-specific syntax from a raw tree-sitter capture to produce the bare
/// module/path token, or `None` if the input cannot be reduced to a single resolvable path.
///
/// The function is deliberately conservative: when a capture contains syntax that implies
/// multiple targets (e.g. Rust brace groups), it returns `None` rather than guessing.
///
/// `ext` is the **lowercased** file extension without the leading dot (e.g. `"rs"`, `"py"`).
///
/// # Language rules
///
/// | Languages | Transformation |
/// |---|---|
/// | Rust (`rs`) | Strip `pub `, `pub(…) `, `use `, `;`; glob `::*` → strip suffix; braces → `None` |
/// | TS/JS/Go/Ruby | Return as-is (already clean from tree-sitter) |
/// | Python (`py`) | Return as-is (captures dotted_name directly) |
/// | Java/Kotlin/Scala | Strip `import `, `static `, `;` |
/// | C/C++ (`c`, `cpp`, `h`, `hpp`, …) | Strip `#include`, `<>`, `""` delimiters |
/// | PHP (`php`) | Strip `use `, `;` |
/// | C# (`cs`) | Strip `using `, `static `, `;` |
/// | Swift (`swift`) | Strip `import ` |
pub fn extract_import_path(raw: &str, ext: &str) -> Option<String> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }

    let result = match ext {
        "rs" => extract_rust(s),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "go" | "rb" | "py" | "pyi" => {
            Some(s.to_string())
        }
        "java" | "kt" | "kts" | "scala" | "sc" => extract_jvm(s),
        "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "hxx" => extract_c(s),
        "php" => extract_php(s),
        "cs" => extract_csharp(s),
        "swift" => extract_swift(s),
        // Unknown extension: pass through trimmed as-is; callers handle the unknown case.
        _ => Some(s.to_string()),
    };

    // Final guard: refuse an empty string after stripping.
    result.filter(|p| !p.trim().is_empty())
}

// ---------------------------------------------------------------------------
// Per-language stripping helpers
// ---------------------------------------------------------------------------

/// Strip Rust import syntax from a raw `use` statement.
///
/// Returns `None` for brace-group imports (`use foo::{bar, baz}`) because they cannot be
/// reduced to a single file path.
fn extract_rust(s: &str) -> Option<String> {
    // Strip optional visibility qualifier.  Handles three forms:
    //   `pub use …`         → plain keyword followed by whitespace
    //   `pub(crate) use …`  → keyword followed by a paren group
    //   (no qualifier)      → pass through
    let s = if let Some(rest) = s.strip_prefix("pub") {
        let rest = rest.trim_start();
        if rest.starts_with('(') {
            // pub(visibility_kind) — skip past the closing ')' and any whitespace.
            match rest.find(')') {
                Some(idx) => rest[idx + 1..].trim_start(),
                None => return None, // malformed: unmatched paren
            }
        } else {
            // plain `pub ` — rest is already trim_start'd
            rest
        }
    } else {
        s
    };

    // Strip `use `.
    let s = if let Some(rest) = s.strip_prefix("use ") {
        rest.trim_start()
    } else {
        s
    };

    // Strip trailing `;`.
    let s = s.trim_end_matches(';').trim();

    // Brace-group imports: cannot resolve to a single file.
    if s.contains('{') {
        return None;
    }

    // Glob import: `crate::foo::*` → `crate::foo`
    let s = if let Some(base) = s.strip_suffix("::*") {
        base
    } else {
        s
    };

    Some(s.to_string())
}

/// Strip JVM-style import syntax (Java, Kotlin, Scala).
///
/// Handles `import static java.util.X;` and plain `import java.util.X`.
fn extract_jvm(s: &str) -> Option<String> {
    let s = if let Some(rest) = s.strip_prefix("import") {
        rest.trim_start()
    } else {
        s
    };
    // Optional `static` keyword (Java static imports).
    let s = strip_prefix_word(s, "static");
    let s = s.trim_end_matches(';').trim();
    Some(s.to_string())
}

/// Strip C/C++ `#include` directive delimiters.
///
/// Handles both `<stdio.h>` (system) and `"myheader.h"` (local) forms.
fn extract_c(s: &str) -> Option<String> {
    let s = s
        .trim_start_matches('#')
        .trim_start()
        .strip_prefix("include")?
        .trim_start();
    // Strip angle-bracket or quote delimiters.
    let inner = if s.starts_with('<') && s.ends_with('>') {
        &s[1..s.len() - 1]
    } else if (s.starts_with('"') && s.ends_with('"'))
        || (s.starts_with('\'') && s.ends_with('\''))
    {
        &s[1..s.len() - 1]
    } else {
        s
    };
    Some(inner.to_string())
}

/// Strip PHP `use` statement syntax.
fn extract_php(s: &str) -> Option<String> {
    let s = if let Some(rest) = s.strip_prefix("use ") {
        rest
    } else {
        s
    };
    Some(s.trim_end_matches(';').trim().to_string())
}

/// Strip C# `using` directive syntax.
///
/// Handles `using static System.Math;` and plain `using System.Linq;`.
fn extract_csharp(s: &str) -> Option<String> {
    let s = if let Some(rest) = s.strip_prefix("using") {
        rest.trim_start()
    } else {
        s
    };
    // Optional `static` keyword.
    let s = strip_prefix_word(s, "static");
    let s = s.trim_end_matches(';').trim();
    Some(s.to_string())
}

/// Strip Swift `import` keyword.
fn extract_swift(s: &str) -> Option<String> {
    let s = if let Some(rest) = s.strip_prefix("import ") {
        rest
    } else {
        s
    };
    Some(s.trim().to_string())
}

// ---------------------------------------------------------------------------
// normalize_path
// ---------------------------------------------------------------------------

/// Normalise a raw path string: convert backslashes to forward slashes and strip a leading `./`.
///
/// Does **not** perform any filesystem resolution — the result is still a logical path token.
pub fn normalize_path(p: &str) -> String {
    let forward = p.replace('\\', "/");
    // Strip a single leading `./` only; `../` is intentional and must survive.
    if let Some(rest) = forward.strip_prefix("./") {
        rest.to_string()
    } else {
        forward
    }
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

/// If `s` begins with `word` (case-sensitive) followed by at least one whitespace character,
/// return the remainder after trimming the leading whitespace.  Otherwise return `s` unchanged.
fn strip_prefix_word<'a>(s: &'a str, word: &str) -> &'a str {
    if let Some(rest) = s.strip_prefix(word) {
        if rest.starts_with(|c: char| c.is_whitespace()) {
            return rest.trim_start();
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Rust ----------------------------------------------------------------

    #[test]
    fn rust_plain_use() {
        assert_eq!(
            extract_import_path("use crate::foo::bar;", "rs"),
            Some("crate::foo::bar".into())
        );
    }

    #[test]
    fn rust_pub_use() {
        assert_eq!(
            extract_import_path("pub use crate::foo;", "rs"),
            Some("crate::foo".into())
        );
    }

    #[test]
    fn rust_pub_crate_use() {
        assert_eq!(
            extract_import_path("pub(crate) use crate::bar;", "rs"),
            Some("crate::bar".into())
        );
    }

    #[test]
    fn rust_glob() {
        assert_eq!(
            extract_import_path("use crate::foo::*;", "rs"),
            Some("crate::foo".into())
        );
    }

    #[test]
    fn rust_braces_returns_none() {
        assert_eq!(extract_import_path("use crate::foo::{bar, baz};", "rs"), None);
    }

    // -- JVM -----------------------------------------------------------------

    #[test]
    fn java_plain() {
        assert_eq!(
            extract_import_path("import java.util.List;", "java"),
            Some("java.util.List".into())
        );
    }

    #[test]
    fn java_static_import() {
        assert_eq!(
            extract_import_path("import static java.util.Collections.sort;", "java"),
            Some("java.util.Collections.sort".into())
        );
    }

    #[test]
    fn kotlin_import() {
        assert_eq!(
            extract_import_path("import kotlin.collections.List", "kt"),
            Some("kotlin.collections.List".into())
        );
    }

    // -- C/C++ ---------------------------------------------------------------

    #[test]
    fn c_system_header() {
        assert_eq!(
            extract_import_path("#include <stdio.h>", "c"),
            Some("stdio.h".into())
        );
    }

    #[test]
    fn c_local_header() {
        assert_eq!(
            extract_import_path("#include \"mylib.h\"", "c"),
            Some("mylib.h".into())
        );
    }

    // -- PHP -----------------------------------------------------------------

    #[test]
    fn php_use() {
        assert_eq!(
            extract_import_path("use Illuminate\\Support\\Facades\\DB;", "php"),
            Some("Illuminate\\Support\\Facades\\DB".into())
        );
    }

    // -- C# ------------------------------------------------------------------

    #[test]
    fn csharp_using() {
        assert_eq!(
            extract_import_path("using System.Linq;", "cs"),
            Some("System.Linq".into())
        );
    }

    #[test]
    fn csharp_static_using() {
        assert_eq!(
            extract_import_path("using static System.Math;", "cs"),
            Some("System.Math".into())
        );
    }

    // -- Swift ---------------------------------------------------------------

    #[test]
    fn swift_import() {
        assert_eq!(
            extract_import_path("import Foundation", "swift"),
            Some("Foundation".into())
        );
    }

    // -- Pass-through languages ----------------------------------------------

    #[test]
    fn typescript_passthrough() {
        assert_eq!(
            extract_import_path("./utils/helpers", "ts"),
            Some("./utils/helpers".into())
        );
    }

    #[test]
    fn python_dotted_name() {
        assert_eq!(
            extract_import_path("os.path", "py"),
            Some("os.path".into())
        );
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(extract_import_path("", "rs"), None);
        assert_eq!(extract_import_path("   ", "ts"), None);
    }

    // -- normalize_path ------------------------------------------------------

    #[test]
    fn normalize_strips_dot_slash() {
        assert_eq!(normalize_path("./foo/bar"), "foo/bar");
    }

    #[test]
    fn normalize_keeps_parent() {
        assert_eq!(normalize_path("../sibling"), "../sibling");
    }

    #[test]
    fn normalize_backslash() {
        assert_eq!(normalize_path("src\\utils\\foo"), "src/utils/foo");
    }
}
