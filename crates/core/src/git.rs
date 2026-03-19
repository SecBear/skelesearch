//! Git integration for diff-aware retrieval.

use std::path::Path;
use std::process::Command;

/// Returns the set of file paths (relative to `repo_root`) that have changed
/// on the current branch compared to the merge base with `main`.
///
/// Falls back to `master` if `main` doesn't exist. Returns an empty set
/// if git is unavailable or the repo has no commits.
pub fn changed_files_on_branch(repo_root: &Path) -> anyhow::Result<std::collections::HashSet<String>> {
    // Try main first, fall back to master
    let base_branch = if branch_exists(repo_root, "main") {
        "main"
    } else if branch_exists(repo_root, "master") {
        "master"
    } else {
        return Ok(std::collections::HashSet::new());
    };

    // Find merge base
    let merge_base = Command::new("git")
        .args(["merge-base", "HEAD", base_branch])
        .current_dir(repo_root)
        .output()?;
    if !merge_base.status.success() {
        // No merge base (e.g., on main itself) — return empty
        return Ok(std::collections::HashSet::new());
    }
    let base = String::from_utf8_lossy(&merge_base.stdout).trim().to_string();

    // Get changed files
    let diff = Command::new("git")
        .args(["diff", "--name-only", &format!("{base}...HEAD")])
        .current_dir(repo_root)
        .output()?;

    let files: std::collections::HashSet<String> = String::from_utf8_lossy(&diff.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();

    tracing::debug!(changed_files = files.len(), base_branch, "branch-scoped file set");
    Ok(files)
}

fn branch_exists(repo_root: &Path, name: &str) -> bool {
    Command::new("git")
        .args(["rev-parse", "--verify", name])
        .current_dir(repo_root)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
