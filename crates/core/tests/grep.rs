// Integration tests for grep_codebase.
//
// Each test uses a fresh TempDir to avoid cross-test interference.

use std::fs;

use skelesearch_core::{grep_codebase, GrepOptions};
use tempfile::TempDir;

/// Helper: write a file relative to `dir`.
fn write_file(dir: &TempDir, rel: &str, content: &str) {
    let path = dir.path().join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

#[test]
fn grep_finds_exact_string() {
    let dir = TempDir::new().unwrap();
    write_file(&dir, "hello.txt", "hello world\nfoo bar\nhello again\n");

    let opts = GrepOptions::default();
    let matches = grep_codebase(dir.path(), "hello", &opts).unwrap();

    assert_eq!(matches.len(), 2, "expected 2 matches for 'hello'");
    assert_eq!(matches[0].line_number, 1);
    assert_eq!(matches[0].line_content, "hello world");
    assert_eq!(matches[1].line_number, 3);
    assert_eq!(matches[1].line_content, "hello again");
}

#[test]
fn grep_finds_regex_pattern() {
    let dir = TempDir::new().unwrap();
    write_file(
        &dir,
        "src/lib.rs",
        "fn foo_bar() {}\nfn baz() {}\nfn qux_bar() {}\n",
    );

    let opts = GrepOptions::default();
    // Matches identifiers ending in _bar.
    let matches = grep_codebase(dir.path(), r"fn \w+_bar", &opts).unwrap();

    assert_eq!(matches.len(), 2);
    assert!(matches[0].line_content.contains("fn foo_bar"));
    assert!(matches[1].line_content.contains("fn qux_bar"));
}

#[test]
fn grep_respects_gitignore() {
    let dir = TempDir::new().unwrap();
    // Create a .git directory so the `ignore` crate treats this as a git repo
    // root and therefore respects the .gitignore file found here.
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    write_file(&dir, ".gitignore", "secret.txt\n");
    write_file(&dir, "secret.txt", "hidden needle\n");
    write_file(&dir, "visible.txt", "visible needle\n");

    let opts = GrepOptions::default();
    let matches = grep_codebase(dir.path(), "needle", &opts).unwrap();

    // Only the visible file should appear; secret.txt is gitignored.
    assert_eq!(matches.len(), 1, "expected gitignored file to be excluded; got: {matches:?}");
    assert!(matches[0].file_path.contains("visible.txt"));
}

#[test]
fn grep_limits_results() {
    let dir = TempDir::new().unwrap();
    // 100 lines, each matching.
    let content: String = (1..=100).map(|i| format!("match line {i}\n")).collect();
    write_file(&dir, "big.txt", &content);

    let opts = GrepOptions { max_results: 5, case_insensitive: false };
    let matches = grep_codebase(dir.path(), "match", &opts).unwrap();

    assert_eq!(matches.len(), 5, "expected result count capped at max_results");
}

#[test]
fn grep_invalid_regex_returns_error() {
    let dir = TempDir::new().unwrap();
    write_file(&dir, "a.txt", "content\n");

    let opts = GrepOptions::default();
    let result = grep_codebase(dir.path(), "[unclosed", &opts);
    assert!(result.is_err(), "invalid regex should return Err");
}

#[test]
fn grep_empty_directory_returns_empty() {
    let dir = TempDir::new().unwrap();
    let opts = GrepOptions::default();
    let matches = grep_codebase(dir.path(), "anything", &opts).unwrap();
    assert!(matches.is_empty());
}

#[test]
fn grep_case_insensitive() {
    let dir = TempDir::new().unwrap();
    write_file(&dir, "code.rs", "let Hello = 1;\nlet HELLO = 2;\nlet world = 3;\n");

    let opts = GrepOptions { max_results: 50, case_insensitive: true };
    let matches = grep_codebase(dir.path(), "hello", &opts).unwrap();

    assert_eq!(matches.len(), 2, "case-insensitive search should match both 'Hello' and 'HELLO'");
}

#[test]
fn grep_matches_unicode_content() {
    let dir = TempDir::new().unwrap();
    // UTF-8 source with non-ASCII identifiers and comments, as you might find
    // in multilingual codebases or documentation-heavy files.
    write_file(
        &dir,
        "unicode_src.py",
        "# αβγ alpha beta gamma\ndef αdd(a, b):\n    return a + b\n# café au lait\n",
    );

    let opts = GrepOptions::default();

    // Match a Greek identifier.
    let matches = grep_codebase(dir.path(), "αdd", &opts).unwrap();
    assert_eq!(matches.len(), 1, "should find the Greek function name");
    assert!(
        matches[0].line_content.contains("αdd"),
        "matched line should contain the Unicode identifier"
    );

    // Match a non-ASCII comment word.
    let accent_matches = grep_codebase(dir.path(), "café", &opts).unwrap();
    assert_eq!(accent_matches.len(), 1, "should find the accented word");
    assert!(accent_matches[0].line_content.contains("café"));
}
