# Structural Graph Layer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a working structural retrieval signal (import graph + def/ref graph + BM25 symbol enrichment) to skelesearch, fixing the currently broken import graph and enabling graph-constrained result expansion.

**Architecture:** Three layers built bottom-up: (1) fix import path resolution so edges resolve to real files, (2) enrich chunk FTS with symbol names for BM25 vocabulary bridging, (3) re-enable graph traversal in the searcher to pull structurally related chunks into results. The import resolver is a trait with per-language heuristic implementations. Symbol enrichment appends normalized symbol names to the existing FTS `normalized` field at index time.

**Tech Stack:** Rust, tree-sitter (0.26, already in workspace), CozoDB (existing storage), `toml` crate (already a dependency for Cargo.toml parsing).

**Research basis:** RepoGraph (ICLR 2025, +32.8% SWE-bench), Aider (PageRank on def/ref graph), Cody/Sourcegraph (BM25F with 5x symbol boost), LocAgent (ACL 2025, 92.7% file localization).

---

## File Structure

```
crates/core/src/
  resolve/              NEW directory — import path resolution
    mod.rs              ImportResolver trait + dispatch + path normalization
    rust.rs             Rust resolver (crate::, self::, super::, workspace)
    typescript.rs       TS/JS resolver (relative, extensions, index files, tsconfig paths)
    python.rs           Python resolver (dotted names, relative imports, __init__.py)
    go.rs               Go resolver (go.mod module path stripping)
  chunker/
    mod.rs              MODIFY — call resolve after extract_edges, enrich normalized with symbols
    languages.rs        MODIFY — fix import_query patterns to capture path sub-nodes (not whole statements)
  indexer.rs            MODIFY — wire resolver into pipeline, pass indexed file set for resolution
  schema.rs             NO CHANGE — EdgeRecord and code_edges already have the right shape
  searcher.rs           MODIFY — re-enable augment_with_graph, add graph score blending
  symbols.rs            NO CHANGE in this phase (existing extraction still runs; enrichment happens in chunker)
  lib.rs                MODIFY — pub mod resolve, re-export ImportResolver

crates/core/tests/
  resolve.rs            NEW — integration tests for import resolution per language
  graph_search.rs       NEW — integration tests for graph-augmented search
```

---

## Chunk 1: Import Path Resolution

### Task 1: ImportResolver trait and dispatch

**Files:**
- Create: `crates/core/src/resolve/mod.rs`
- Modify: `crates/core/src/lib.rs` (add `pub mod resolve`)

- [ ] **Step 1: Create resolve module with trait and dispatch**

```rust
// crates/core/src/resolve/mod.rs
mod rust;
mod typescript;
mod python;
mod go;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Resolves a raw import string (as captured by tree-sitter) to a file path
/// relative to the project root.
///
/// Implementations are heuristic — they resolve 85-95% of within-project
/// imports using filesystem conventions, without requiring compilation.
pub trait ImportResolver {
    /// Attempt to resolve `raw_import` (the tree-sitter capture text from an
    /// import statement) to a project-relative file path.
    ///
    /// - `from_file`: project-relative path of the importing file
    /// - `project_root`: absolute path to the project root
    /// - `indexed_files`: set of all project-relative file paths known to the index
    ///
    /// Returns `None` if the import cannot be resolved (external dep, ambiguous, etc.)
    fn resolve(
        &self,
        raw_import: &str,
        from_file: &Path,
        project_root: &Path,
        indexed_files: &HashSet<String>,
    ) -> Option<String>;
}

/// Select the appropriate resolver for a file extension.
pub fn resolver_for_extension(ext: &str) -> Option<Box<dyn ImportResolver>> {
    match ext {
        "rs" => Some(Box::new(rust::RustResolver)),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => {
            Some(Box::new(typescript::TsResolver))
        }
        "py" => Some(Box::new(python::PythonResolver)),
        "go" => Some(Box::new(go::GoResolver)),
        _ => None,
    }
}

/// Extract the module/path portion from raw tree-sitter import capture text.
///
/// Many import_query patterns capture the entire statement node (e.g.,
/// `use crate::foo::bar;` for Rust, `import java.util.List;` for Java).
/// This function extracts just the path/module portion.
pub fn extract_import_path(raw: &str, ext: &str) -> Option<String> {
    let trimmed = raw.trim();
    match ext {
        // Already clean from tree-sitter (string literal, quotes stripped)
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "go" | "rb" => {
            Some(trimmed.to_string())
        }
        "rs" => {
            // "use crate::foo::bar;" or "use crate::foo::bar"
            let s = trimmed.strip_prefix("use ")?.trim_end_matches(';').trim();
            // Strip visibility: "pub use ..." or "pub(crate) use ..."
            // Already handled since we strip "use " prefix from the full statement
            Some(s.to_string())
        }
        "py" => {
            // Already captures dotted_name only (e.g., "os.path")
            Some(trimmed.to_string())
        }
        "java" | "kt" | "scala" => {
            // "import java.util.List;" → "java.util.List"
            let s = trimmed
                .strip_prefix("import ")?
                .trim_start_matches("static ")
                .trim_end_matches(';')
                .trim();
            Some(s.to_string())
        }
        "c" | "cpp" | "cc" | "cxx" | "h" | "hpp" => {
            // "#include <stdio.h>" or "#include \"myheader.h\""
            let s = trimmed.strip_prefix("#include")?.trim();
            let path = s
                .trim_start_matches(|c| c == '<' || c == '"')
                .trim_end_matches(|c| c == '>' || c == '"')
                .trim();
            Some(path.to_string())
        }
        "php" => {
            // "use App\Models\User;" → "App\Models\User"
            let s = trimmed
                .strip_prefix("use ")?
                .trim_end_matches(';')
                .trim();
            Some(s.to_string())
        }
        "cs" => {
            // "using System.Linq;" → "System.Linq"
            let s = trimmed
                .strip_prefix("using ")?
                .trim_start_matches("static ")
                .trim_end_matches(';')
                .trim();
            Some(s.to_string())
        }
        "swift" => {
            // "import Foundation" → "Foundation"
            let s = trimmed.strip_prefix("import ")?.trim();
            Some(s.to_string())
        }
        _ => Some(trimmed.to_string()),
    }
}

/// Normalize a project-relative path: forward slashes, no leading "./".
pub fn normalize_path(p: &str) -> String {
    p.replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}
```

- [ ] **Step 2: Wire into lib.rs**

Add `pub mod resolve;` and `pub use resolve::{ImportResolver, resolver_for_extension, extract_import_path};` to `crates/core/src/lib.rs`.

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p skelesearch-core --features storage-sqlite`

### Task 2: Rust import resolver

**Files:**
- Create: `crates/core/src/resolve/rust.rs`

- [ ] **Step 1: Implement RustResolver**

```rust
// crates/core/src/resolve/rust.rs
use super::{ImportResolver, normalize_path};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct RustResolver;

impl ImportResolver for RustResolver {
    fn resolve(
        &self,
        raw_import: &str,
        from_file: &Path,
        project_root: &Path,
        indexed_files: &HashSet<String>,
    ) -> Option<String> {
        let path = super::extract_import_path(raw_import, "rs")?;

        // Handle "crate::a::b::c", "self::x", "super::x"
        let segments: Vec<&str> = path.split("::").collect();
        if segments.is_empty() {
            return None;
        }

        // Find the crate root relative to from_file
        let from_dir = from_file.parent()?;
        let crate_root = find_crate_src_dir(from_file, project_root)?;

        let resolved_segments = match segments[0] {
            "crate" => {
                // Resolve from crate src directory
                resolve_from_base(&crate_root, &segments[1..], project_root, indexed_files)
            }
            "self" => {
                // Resolve from current module's directory
                let base = module_dir(from_file);
                resolve_from_base(&base, &segments[1..], project_root, indexed_files)
            }
            "super" => {
                let mut base = module_dir(from_file);
                let mut remaining = &segments[1..];
                // Count consecutive "super" segments
                while remaining.first() == Some(&"super") {
                    base = base.parent()?.to_path_buf();
                    remaining = &remaining[1..];
                }
                // Initial "super" already consumed
                base = base.parent()?.to_path_buf();
                resolve_from_base(&base, remaining, project_root, indexed_files)
            }
            _ => {
                // External crate or unqualified — skip for now.
                // Could look up workspace Cargo.toml in the future.
                None
            }
        };

        resolved_segments
    }
}

/// Given path segments like ["a", "b", "c"], try:
/// 1. base/a/b/c.rs  (all segments are modules)
/// 2. base/a/b/c/mod.rs
/// 3. base/a/b.rs  (c is an item in module b)
/// 4. base/a/b/mod.rs
fn resolve_from_base(
    base: &Path,
    segments: &[&str],
    project_root: &Path,
    indexed_files: &HashSet<String>,
) -> Option<String> {
    if segments.is_empty() {
        return None;
    }

    // Try all segments as modules
    let mut path = base.to_path_buf();
    for seg in segments {
        path.push(seg);
    }

    // Try path.rs
    let candidate = path.with_extension("rs");
    if let Some(rel) = check_exists(&candidate, project_root, indexed_files) {
        return Some(rel);
    }

    // Try path/mod.rs
    let candidate = path.join("mod.rs");
    if let Some(rel) = check_exists(&candidate, project_root, indexed_files) {
        return Some(rel);
    }

    // Try dropping the last segment (it might be an item, not a module)
    if segments.len() > 1 {
        let parent_segments = &segments[..segments.len() - 1];
        let mut path = base.to_path_buf();
        for seg in parent_segments {
            path.push(seg);
        }

        let candidate = path.with_extension("rs");
        if let Some(rel) = check_exists(&candidate, project_root, indexed_files) {
            return Some(rel);
        }

        let candidate = path.join("mod.rs");
        if let Some(rel) = check_exists(&candidate, project_root, indexed_files) {
            return Some(rel);
        }
    }

    None
}

fn check_exists(
    abs_path: &Path,
    project_root: &Path,
    indexed_files: &HashSet<String>,
) -> Option<String> {
    let rel = abs_path.strip_prefix(project_root).ok()?;
    let rel_str = normalize_path(&rel.to_string_lossy());
    if indexed_files.contains(&rel_str) {
        Some(rel_str)
    } else {
        None
    }
}

/// Find the src/ directory of the crate containing `file`.
/// Walks up from the file looking for Cargo.toml, then returns its parent + "src/".
fn find_crate_src_dir(file: &Path, project_root: &Path) -> Option<PathBuf> {
    let abs_file = if file.is_absolute() {
        file.to_path_buf()
    } else {
        project_root.join(file)
    };

    let mut dir = abs_file.parent()?;
    loop {
        if dir.join("Cargo.toml").exists() {
            return Some(dir.join("src"));
        }
        if dir == project_root || dir.parent().is_none() {
            // Fallback: assume src/ is at project root
            return Some(project_root.join("src"));
        }
        dir = dir.parent()?;
    }
}

/// The directory representing the current module.
/// For src/foo/bar.rs → src/foo/
/// For src/foo/mod.rs → src/foo/
fn module_dir(file: &Path) -> PathBuf {
    let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if file_name == "mod.rs" || file_name == "lib.rs" || file_name == "main.rs" {
        file.parent().unwrap_or(file).to_path_buf()
    } else {
        file.parent().unwrap_or(file).to_path_buf()
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p skelesearch-core --features storage-sqlite`

### Task 3: TypeScript/JavaScript import resolver

**Files:**
- Create: `crates/core/src/resolve/typescript.rs`

- [ ] **Step 1: Implement TsResolver**

Key resolution rules:
- Relative paths (`./foo`, `../bar`): resolve from importing file's directory
- Try extensions in order: exact → `.ts` → `.tsx` → `.js` → `.jsx`
- Try directory index: `{path}/index.ts` → `{path}/index.tsx` → `{path}/index.js`
- JS→TS extension rewriting: if `./foo.js` doesn't exist but `./foo.ts` does, prefer `.ts`
- Bare specifiers (`react`, `lodash`): skip — external deps, not in-repo
- If it starts with `@` or `~` or `#`: skip — alias, would need tsconfig parsing

The implementation should be ~100 lines following the same pattern as `RustResolver`.

- [ ] **Step 2: Verify it compiles**

### Task 4: Python import resolver

**Files:**
- Create: `crates/core/src/resolve/python.rs`

- [ ] **Step 1: Implement PythonResolver**

Key resolution rules:
- Dots become path separators: `foo.bar.baz` → try `foo/bar/baz.py`, `foo/bar/baz/__init__.py`, `foo/bar.py`
- Relative imports (leading dots): count dots, walk up from importing file's package directory
- Try `src/` prefix for src-layout projects
- Skip imports not found in project tree (third-party)

~80 lines.

- [ ] **Step 2: Verify it compiles**

### Task 5: Go import resolver

**Files:**
- Create: `crates/core/src/resolve/go.rs`

- [ ] **Step 1: Implement GoResolver**

Key resolution rules:
- Parse `go.mod` for module path (e.g., `github.com/user/repo`)
- Strip module path prefix from import path → directory
- All `.go` files in directory belong to the package
- Return the directory path (any file in it is a valid resolution target)

~60 lines.

- [ ] **Step 2: Verify it compiles**

### Task 6: Integration tests for resolvers

**Files:**
- Create: `crates/core/tests/resolve.rs`
- Uses: `crates/core/tests/fixtures/sample_repo/` (existing fixture, may need extensions)

- [ ] **Step 1: Write tests for each resolver**

Test each resolver against fixture files. At minimum:
- Rust: `crate::x::y` → `src/x/y.rs`, `self::z` → sibling, item-in-module fallback
- TS: `./utils` → `utils.ts`, `../shared` → `../shared/index.ts`, bare specifier → None
- Python: `foo.bar` → `foo/bar.py` or `foo/bar/__init__.py`, relative import
- Go: module-path stripping, directory resolution

- [ ] **Step 2: Run tests**

Run: `cargo test -p skelesearch-core --test resolve --features storage-sqlite`
Expected: All pass.

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: import path resolver trait + implementations for Rust/TS/Python/Go

ImportResolver trait with heuristic per-language implementations:
- Rust: crate::/self::/super:: resolution via Cargo.toml lookup
- TS/JS: relative paths with extension/index probing, JS→TS rewriting
- Python: dotted names to paths, relative imports, src-layout
- Go: go.mod module path stripping to directory

Expected accuracy: 85-95% for within-project imports."
```

---

## Chunk 2: Wire Resolver into Indexer + Symbol Enrichment

### Task 7: Fix import_query patterns to capture path sub-nodes

**Files:**
- Modify: `crates/core/src/chunker/languages.rs`

- [ ] **Step 1: Update import queries that capture entire statements**

For languages where import_query captures the whole node, refine to capture the path sub-node. The key languages to fix:

- **Rust**: `(use_declaration path: (scoped_identifier) @path)` OR keep `(use_declaration) @path` and rely on `extract_import_path()` to strip `use` prefix (simpler, more robust). **Decision: keep existing query, use extract_import_path.** The tree-sitter Rust grammar's `use_declaration` doesn't always have a simple `path` field due to `use crate::{a, b}` syntax. Stripping in Rust code is safer.
- **Java/Kotlin/C#/PHP/Swift/C/C++**: Same decision — keep queries, rely on `extract_import_path()`.

This means: **no changes to languages.rs in this task.** The `extract_import_path()` function in `resolve/mod.rs` handles the stripping for all languages. The import_query patterns are correct for what they capture; the fix is in the resolution step, not the capture step.

- [ ] **Step 2: Verify no regressions**

Run: `cargo test -p skelesearch-core --test chunker --features storage-sqlite`

### Task 8: Wire resolver into indexer pipeline

**Files:**
- Modify: `crates/core/src/indexer.rs`

- [ ] **Step 1: Add resolver invocation between edge extraction and storage**

In the indexer's `index_path` method, after `let edges = chunker.extract_edges(...)`, resolve each edge's `to_file` using the appropriate resolver. The `indexed_files` set can be built from the current walk + already-indexed files.

Key changes to `indexer.rs`:
1. After the file walk collects all `FileCandidate`s, build a `HashSet<String>` of all known project-relative paths (both from the current walk and from `backend.list_indexed_paths()`)
2. In the batch loop, after `extract_edges()`, run each edge through the resolver:
   ```rust
   let ext = Path::new(&fc.rel_path).extension().and_then(|e| e.to_str()).unwrap_or("");
   let resolved_edges: Vec<ImportEdge> = edges.into_iter().filter_map(|e| {
       let resolver = resolver_for_extension(ext)?;
       let clean = extract_import_path(&e.to_file, ext)?;
       let resolved = resolver.resolve(&clean, Path::new(&fc.rel_path), &root, &all_files)?;
       Some(ImportEdge { from_file: e.from_file, to_file: resolved })
   }).collect();
   ```
3. Edges that don't resolve are silently dropped (external deps, ambiguous)
4. Store the `project_root` as a field on the `Indexer` struct (it's passed to `index_path` already)

- [ ] **Step 2: Run existing tests**

Run: `cargo test -p skelesearch-core --features storage-sqlite`
Expected: All pass. Indexer tests should still work; resolved edges are a superset improvement.

### Task 9: Enrich chunk FTS normalized field with symbol names

**Files:**
- Modify: `crates/core/src/indexer.rs`

- [ ] **Step 1: Append symbol names to chunk normalized text**

In the indexer, after `extract_symbols()` runs and before `upsert_chunks()`, for each chunk in the batch, find overlapping symbols (by line range) and append their normalized names to the chunk's `normalized` field.

```rust
// After symbols are extracted, before chunk upsert:
for chunk_record in &mut chunk_records {
    let overlapping_symbols: Vec<&SymbolDef> = bf.symbols.iter()
        .filter(|s| s.start_line <= chunk_record.end_line && s.end_line >= chunk_record.start_line)
        .collect();
    if !overlapping_symbols.is_empty() {
        let symbol_text: String = overlapping_symbols.iter()
            .map(|s| format!("{} {}", normalize_for_fts(&s.name), s.kind))
            .collect::<Vec<_>>()
            .join(" ");
        chunk_record.normalized.push(' ');
        chunk_record.normalized.push_str(&symbol_text);
    }
}
```

This means a chunk containing `struct StateStore { ... }` will have `state store struct` appended to its FTS-indexed text, making it findable by BM25 queries mentioning "state store" even when the chunk's code text uses different vocabulary.

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p skelesearch-core --features storage-sqlite`

- [ ] **Step 3: Write a test**

Add a test to `crates/core/tests/indexer.rs` (or a new file) that indexes a small fixture, then verifies the stored chunk's normalized text contains the expected symbol names.

- [ ] **Step 4: Commit**

```
git add -A && git commit -m "feat: wire import resolver + symbol BM25 enrichment into indexer

- Import edges now resolve to real file paths via heuristic resolvers
- Unresolved edges (external deps) are silently dropped
- Symbol names appended to chunk FTS normalized field for BM25 bridging
- e.g., chunk defining 'struct StateStore' now has 'state store struct'
  in its FTS index, enabling vocabulary-gap retrieval"
```

---

## Chunk 3: Re-enable Graph Traversal in Searcher

### Task 10: Re-enable augment_with_graph

**Files:**
- Modify: `crates/core/src/searcher.rs`

- [ ] **Step 1: Un-disable graph augmentation**

In `searcher.rs`, the graph augmentation is currently disabled at lines ~162-168:
```rust
// The include_graph parameter is accepted but currently a no-op.
let _ = (include_graph, max_depth);
```

Replace with the actual call to `augment_with_graph`:
```rust
if include_graph && max_depth > 0 {
    hits = self.augment_with_graph(hits, max_depth).await?;
}
```

The existing `augment_with_graph` method does BFS over `traverse_imports`, pulls chunks from reachable files, and assigns `score: 0.0` with `why: "graph"`. This will now work because edges contain resolved file paths.

- [ ] **Step 2: Improve graph result scoring**

Currently `augment_with_graph` assigns `score: 0.0` to graph-expanded results. This pushes them to the bottom. Instead, assign a decayed score based on BFS depth and the score of the originating hit:

```rust
// In augment_with_graph, when creating graph-expanded SearchResults:
let base_score = originating_hit_score * 0.5_f64.powi(depth as i32);
SearchResult {
    score: base_score,
    match_quality: "graph".to_string(),
    why: format!("graph (depth {})", depth),
    ..
}
```

This requires modifying `augment_with_graph` to track which hit triggered each expansion and at what depth. The existing BFS structure already tracks depth via the loop counter.

- [ ] **Step 3: Position graph augmentation correctly in the pipeline**

Graph augmentation should happen AFTER hybrid_search (so we have initial hits to expand from) but BEFORE MMR reranking and cross-encoder reranking (so expanded results participate in quality filtering).

Current pipeline order:
1. LLM expansion → keyword extraction → embed → hybrid_search → label_match_quality
2. ~~graph augmentation (disabled)~~
3. MMR → reranker → token budget

Move graph augmentation to position 2 (exactly where the disabled code already sits). The code placement is already correct; just un-comment it.

- [ ] **Step 4: Run tests**

Run: `cargo test -p skelesearch-core --features storage-sqlite`

### Task 11: Integration test for graph-augmented search

**Files:**
- Create: `crates/core/tests/graph_search.rs`

- [ ] **Step 1: Write integration test**

Create a test that:
1. Sets up a mock backend with 3 files: A imports B, B imports C
2. Indexes them so edges are stored
3. Searches for content in file A with `include_graph=true, max_depth=1`
4. Verifies that chunks from file B appear in results (graph expansion)
5. Verifies that chunks from file C do NOT appear (depth=1, not depth=2)

Use the existing `test_utils.rs` patterns and `CountingTestProvider` or a simple mock.

- [ ] **Step 2: Run tests**

Run: `cargo test -p skelesearch-core --test graph_search --features storage-sqlite`
Expected: PASS

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: re-enable graph-augmented search with depth-decayed scoring

- Import graph traversal now works (edges resolve to real file paths)
- Graph-expanded results scored with 0.5^depth decay from originating hit
- Positioned after hybrid_search, before MMR + reranker
- CLI --graph flag and MCP include_graph field now functional"
```

---

## Chunk 4: Eval and Measure

### Task 12: Re-index skelegent and run eval

**Files:**
- No code changes

- [ ] **Step 1: Build release binary**

```bash
cargo build --release --features storage-sqlite
```

- [ ] **Step 2: Delete old index and re-index with resolved edges**

```bash
rm -rf /Users/bear/dev/golden-neuron/skelegent/.skelesearch/
# Source .env for VOYAGE_API_KEY + OPENAI_API_KEY
set -a && . .env && set +a
./target/release/skelesearch index /Users/bear/dev/golden-neuron/skelegent --provider voyage
```

Verify: output should show indexed files/chunks AND the number of resolved import edges should be logged (add a tracing::info! in the indexer if not already present).

- [ ] **Step 3: Run eval without graph (baseline with symbol enrichment only)**

```bash
./target/release/skelesearch eval eval/skelegent-eval.json --provider voyage --json
```

Compare R@5/R@10/MRR against previous baseline (0.700/0.700/0.606). Symbol enrichment should improve scores on vocabulary-gap queries.

- [ ] **Step 4: Run search with --graph to test graph expansion manually**

```bash
./target/release/skelesearch search "how do agents remember things across conversations" \
  --provider voyage --top-k 10 --graph -vv
```

Check if `layer0/src/state.rs` or `layer0/src/effect.rs` now appears in results via graph expansion.

- [ ] **Step 5: Document results**

Update the eval baseline in the handoff context. Record:
- Aggregate R@5, R@10, MRR with symbol enrichment (no graph)
- Per-case results for the 5 previously failing queries
- Any improvements from graph expansion on the 2 zero-recall queries

- [ ] **Step 6: Commit eval results**

```
git add -A && git commit -m "eval: re-run with resolved import graph + symbol BM25 enrichment

Results: [fill in actual numbers]
Previous: R@5=0.700 R@10=0.700 MRR=0.606"
```

---

## Deferred (not in this plan, future work)

These are documented for prioritization after measuring the impact of this plan:

1. **tree-sitter tags.scm integration** — Replace `extract_symbols` with `tree-sitter-tags` crate for richer def/ref extraction. Enables def→ref graph edges beyond imports. Blocked on verifying `tree-sitter-tags` v0.26 compatibility.
2. **PageRank on file-level graph** — Compute PageRank at index time from resolved import edges + def/ref edges. Use as a score boost in retrieval. Requires the def/ref graph from item 1.
3. **Git recency boost** — `git log` recent commit count per file as a ranking signal.
4. **Co-change index** — Mine `git log` for files that change together, use as implicit coupling edges.
5. **Contextual chunk descriptions** — LLM-generated per-chunk summaries for embedding (Anthropic-style).
