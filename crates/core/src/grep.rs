// grep.rs — file-walking regex search over a codebase directory.
//
// Uses the `ignore` crate's WalkBuilder so .gitignore rules are respected
// automatically.  Binary files are silently skipped: read_to_string fails on
// non-UTF-8 content and the error is eaten.

use std::path::Path;

use anyhow::Context as _;
use ignore::WalkBuilder;
use regex::Regex;

/// A single line that matched the search pattern.
#[derive(Debug, Clone)]
pub struct GrepMatch {
    /// Path relative to the search root.
    pub file_path: String,
    /// 1-indexed line number.
    pub line_number: usize,
    /// Full text of the matching line.
    pub line_content: String,
}

/// Tunable parameters for `grep_codebase`.
#[derive(Debug, Clone)]
pub struct GrepOptions {
    /// Maximum number of matches returned across all files.
    pub max_results: usize,
    /// When true, the regex is compiled with the `(?i)` flag.
    pub case_insensitive: bool,
}

impl Default for GrepOptions {
    fn default() -> Self {
        Self { max_results: 50, case_insensitive: false }
    }
}

/// Walk `root` respecting .gitignore and return every line matching `pattern`.
///
/// # Errors
/// Returns an error if `pattern` is not a valid regex.  I/O errors on
/// individual files (binary files, permission errors) are silently skipped so
/// a single unreadable file never aborts a whole-codebase scan.
pub fn grep_codebase(root: &Path, pattern: &str, opts: &GrepOptions) -> anyhow::Result<Vec<GrepMatch>> {
    // Compile once — reject bad patterns immediately with a clear error.
    let re = if opts.case_insensitive {
        Regex::new(&format!("(?i){pattern}")).with_context(|| format!("invalid regex pattern: {pattern}"))?
    } else {
        Regex::new(pattern).with_context(|| format!("invalid regex pattern: {pattern}"))?
    };

    let mut results = Vec::new();

    let walker = WalkBuilder::new(root).build();
    for entry in walker {
        if results.len() >= opts.max_results {
            break;
        }
        let entry = entry?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }

        // Binary files produce a UTF-8 decode error — skip them silently.
        let content = match std::fs::read_to_string(entry.path()) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Strip the root prefix so callers get relative paths.
        let rel_path = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .into_owned();

        for (i, line) in content.lines().enumerate() {
            if results.len() >= opts.max_results {
                break;
            }
            if re.is_match(line) {
                results.push(GrepMatch {
                    file_path: rel_path.clone(),
                    line_number: i + 1,
                    line_content: line.to_string(),
                });
            }
        }
    }

    Ok(results)
}
