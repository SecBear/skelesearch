//! Co-change analysis: mine git history to find files that frequently
//! change together.  Results are stored as `cochange_edges` in CozoDB and
//! can later be used as a graph signal during retrieval.
//!
//! The algorithm:
//! 1. Run `git log --name-only --format='%H' --no-merges -N` to get the
//!    commit/file list for the last N commits.
//! 2. Parse each commit's file set.
//! 3. Filter to source files only (same extensions the indexer uses).
//! 4. Cap per-commit file count at `MAX_FILES_PER_COMMIT` to avoid O(N²)
//!    pair explosion on large omnibus commits.
//! 5. For every unordered pair (a, b) appearing in the same commit, increment
//!    a co-change counter.  Pairs are canonicalized as (min, max) to avoid
//!    double-counting.
//! 6. Compute Jaccard similarity: |A∩B| / |A∪B|.
//! 7. Return pairs where co_count ≥ min_commits AND jaccard ≥ 0.1.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use crate::indexer::default_include_extensions;

/// Maximum files from a single commit that participate in pair generation.
/// Omnibus commits touching hundreds of files create O(N²) pairs and carry
/// little structural signal; capping at 20 keeps worst-case pairs ≤ 190.
const MAX_FILES_PER_COMMIT: usize = 20;

/// A pair of files that co-change frequently in git history.
#[derive(Debug, Clone, PartialEq)]
pub struct CoChangePair {
    /// Lexicographically smaller of the two file paths.
    pub file_a: String,
    /// Lexicographically larger of the two file paths.
    pub file_b: String,
    /// Number of commits where both files appear.
    pub cochange_count: usize,
    /// Jaccard similarity: cochange_count / (commits_a + commits_b - cochange_count).
    pub jaccard: f64,
}

/// Analyze git history to find files that frequently change together.
///
/// # Arguments
/// * `repo_path`   – root of the git repository.
/// * `min_commits` – minimum co-change count to include a pair (default: 3).
/// * `max_commits` – how many recent commits to inspect (default: 500).
///
/// Returns an empty `Vec` (not an error) when git is unavailable, the
/// directory is not a git repo, or the history is empty.
pub fn compute_cochange_pairs(
    repo_path: &Path,
    min_commits: usize,
    max_commits: usize,
) -> anyhow::Result<Vec<CoChangePair>> {
    let source_exts = default_include_extensions();

    // `git log --name-only --format='%H' --no-merges -<N>` produces blocks like:
    //
    //   <40-char-sha>
    //
    //   src/foo.rs
    //   src/bar.rs
    //
    //   <next-sha>
    //   ...
    //
    // (blank lines separate commits when using --name-only)
    let output = Command::new("git")
        .args([
            "log",
            "--name-only",
            "--format=%H",
            "--no-merges",
            &format!("-{max_commits}"),
        ])
        .current_dir(repo_path)
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        // git not available or not a repo — skip silently.
        Ok(_) | Err(_) => return Ok(vec![]),
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let commits = parse_git_log(&text, &source_exts);

    if commits.is_empty() {
        return Ok(vec![]);
    }

    // per-file commit counts, used for Jaccard denominator
    let mut file_counts: HashMap<String, usize> = HashMap::new();
    // co-change counts for canonical (a, b) pairs where a < b
    let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();

    for files in &commits {
        // Update per-file counts.
        for f in files {
            *file_counts.entry(f.clone()).or_insert(0) += 1;
        }
        // Enumerate unordered pairs.  `files` is already capped at
        // MAX_FILES_PER_COMMIT and deduplicated by `parse_git_log`.
        let file_list: Vec<&String> = files.iter().collect();
        for i in 0..file_list.len() {
            for j in (i + 1)..file_list.len() {
                let (a, b) = normalize_pair(file_list[i], file_list[j]);
                *pair_counts.entry((a, b)).or_insert(0) += 1;
            }
        }
    }

    let mut result: Vec<CoChangePair> = pair_counts
        .into_iter()
        .filter_map(|((file_a, file_b), co)| {
            if co < min_commits {
                return None;
            }
            let a_count = *file_counts.get(&file_a).unwrap_or(&0);
            let b_count = *file_counts.get(&file_b).unwrap_or(&0);
            // Jaccard: |A∩B| / |A∪B|  where |A∪B| = a + b - co
            let union = a_count + b_count - co;
            if union == 0 {
                return None;
            }
            let jaccard = co as f64 / union as f64;
            if jaccard < 0.1 {
                return None;
            }
            Some(CoChangePair { file_a, file_b, cochange_count: co, jaccard })
        })
        .collect();

    // Stable order makes test output deterministic.
    result.sort_by(|a, b| a.file_a.cmp(&b.file_a).then(a.file_b.cmp(&b.file_b)));
    Ok(result)
}

/// Parse `git log --name-only --format='%H'` output into a list of file sets.
///
/// Each inner `Vec` holds the (filtered, capped) source files touched by one
/// commit.  Commits whose file set is empty after filtering are dropped.
fn parse_git_log(text: &str, source_exts: &HashSet<String>) -> Vec<Vec<String>> {
    let mut commits: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    // Track whether we're inside a file-listing block (i.e. we have seen
    // the SHA line for the current commit).
    let mut in_commit = false;

    for line in text.lines() {
        let line = line.trim();
        if is_commit_sha(line) {
            // Flush the previous commit's file list.
            if in_commit && !current.is_empty() {
                commits.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
            in_commit = true;
            continue;
        }
        if !in_commit || line.is_empty() {
            continue;
        }
        // Skip files whose extension is not in the source allowlist.
        if !is_source_file(line, source_exts) {
            continue;
        }
        // Cap per-commit participation to avoid O(N²) pair explosion.
        if current.len() >= MAX_FILES_PER_COMMIT {
            continue;
        }
        current.push(line.to_string());
    }
    // Flush the last commit.
    if in_commit && !current.is_empty() {
        commits.push(current);
    }
    commits
}

/// A line is a commit SHA if it consists of exactly 40 lowercase hex chars.
fn is_commit_sha(line: &str) -> bool {
    line.len() == 40 && line.chars().all(|c| c.is_ascii_hexdigit())
}

/// Return `true` if the file's extension is in `source_exts`.
fn is_source_file(path: &str, source_exts: &HashSet<String>) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| source_exts.contains(&e.to_lowercase()))
        .unwrap_or(false)
}

/// Canonicalize a pair so that `file_a < file_b` lexicographically.
/// This prevents double-counting (a, b) and (b, a) as separate pairs.
pub(crate) fn normalize_pair(x: &str, y: &str) -> (String, String) {
    if x <= y {
        (x.to_string(), y.to_string())
    } else {
        (y.to_string(), x.to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_pair_canonical_order() {
        let (a, b) = normalize_pair("src/foo.rs", "src/bar.rs");
        assert_eq!(a, "src/bar.rs");
        assert_eq!(b, "src/foo.rs");
        assert!(a <= b, "file_a must be ≤ file_b");
    }

    #[test]
    fn normalize_pair_already_ordered() {
        let (a, b) = normalize_pair("alpha.rs", "zeta.rs");
        assert_eq!(a, "alpha.rs");
        assert_eq!(b, "zeta.rs");
    }

    #[test]
    fn normalize_pair_equal_inputs() {
        let (a, b) = normalize_pair("same.rs", "same.rs");
        assert_eq!(a, b);
    }

    #[test]
    fn jaccard_calculation() {
        // Manually construct the scenario:
        //   A appears in 5 commits, B in 5, co-change in 4.
        //   Jaccard = 4 / (5 + 5 - 4) = 4/6 ≈ 0.667
        let co = 4usize;
        let a_count = 5usize;
        let b_count = 5usize;
        let union = a_count + b_count - co;
        let jaccard = co as f64 / union as f64;
        assert!((jaccard - 0.6666667).abs() < 1e-5);
    }

    #[test]
    fn parse_git_log_basic() {
        let source_exts = default_include_extensions();
        let log = "\
a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2\n\
\n\
src/foo.rs\n\
src/bar.rs\n\
\n\
deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n\
\n\
src/foo.rs\n\
src/baz.rs\n\
";
        let commits = parse_git_log(log, &source_exts);
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[0], vec!["src/foo.rs", "src/bar.rs"]);
        assert_eq!(commits[1], vec!["src/foo.rs", "src/baz.rs"]);
    }

    #[test]
    fn parse_git_log_filters_binary() {
        let source_exts = default_include_extensions();
        let log = "\
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\
\n\
assets/image.png\n\
src/main.rs\n\
";
        let commits = parse_git_log(log, &source_exts);
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0], vec!["src/main.rs"]);
    }

    #[test]
    fn parse_git_log_caps_per_commit() {
        let source_exts = default_include_extensions();
        // 25 .rs files in a single commit — should be capped at MAX_FILES_PER_COMMIT (20).
        let files: String = (0..25).map(|i| format!("src/file{i}.rs\n")).collect();
        let log = format!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n{files}");
        let commits = parse_git_log(&log, &source_exts);
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].len(), MAX_FILES_PER_COMMIT);
    }

    #[test]
    fn parse_git_log_empty_or_no_source_files() {
        let source_exts = default_include_extensions();
        let log = "\
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\
\n\
README.bin\n\
photo.jpg\n\
";
        // All filtered out → commit should not appear in results.
        let commits = parse_git_log(log, &source_exts);
        assert_eq!(commits.len(), 0);
    }

    #[test]
    fn is_commit_sha_valid() {
        assert!(is_commit_sha("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"));
        assert!(!is_commit_sha("not-a-sha"));
        assert!(!is_commit_sha("")); // too short
        assert!(!is_commit_sha(
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2x"
        )); // 41 chars
    }
}
