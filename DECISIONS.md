# Architectural Decisions

This file records significant architectural choices made during design, with rationale.
Read this before modifying core abstractions or switching dependencies.

---

## ADR-001: Use CozoDB directly (not the CozoDB state management crate)

**Decision:** Depend on `cozo` crate directly; do not use the CozoDB state management crate's StateStore abstraction from the extension crate.

**Why:** The CozoDB state management crate hardcodes 1536-dim embeddings in its schema. skelesearch needs
configurable dimensionality to support fastembed models (768-dim jina-v2-base-code) alongside
OpenAI (1536-dim) and others. A code-search-specific schema (`chunks`, `files`, `code_edges`)
is also cleaner than the generic KV model in StateStore.

**Trade-off:** Some duplication with the CozoDB state management crate's RRF implementation. Acceptable — our
schema and retrieval logic is small and tailored to code search.

---

## ADR-002: CozoDB over LanceDB / sqlite-vec

**Decision:** Use CozoDB as the embedded storage and search engine.

**Why:** CozoDB is the only embedded database that provides HNSW vector search, BM25 full-text
search, AND graph traversal (recursive Datalog) natively in one query. The import/call graph
is a first-class differentiator of skelesearch — it requires the graph layer to be collocated
with the vector index, not a separate system.

**Risk:** CozoDB's last release was v0.7.6 (December 2023). Development appears stalled.
It is pre-1.0 with no stability guarantees between versions.

**Mitigation:** All CozoDB code is isolated in `crates/core/src/schema.rs` behind a
`StorageBackend` trait. Migrating to LanceDB+Tantivy is a single-file change. v0.7.6 is
feature-complete for our use case — we don't need upstream to evolve.

**Alternatives considered:** LanceDB (active, no graph), sqlite-vec (simplest, IVF not HNSW,
no graph), Qdrant (requires a server process).

---

## ADR-003: No SPLADE in v1

**Decision:** The sparse retrieval leg uses CozoDB FTS (BM25) only. SPLADE is deferred to v2+.

**Why:**
- No code-domain benchmarks for SPLADE exist (CoIR benchmark covers dense models only)
- Both fastembed-rs SPLADE models (SPLADEPPV1, BGEM3-sparse) are trained on English NLP,
  not code — vocabulary expansion may harm code identifier matching
- SPLADE is 24× slower than BM25 at query time on CPU
- CozoDB has no native inverted index for SPLADE storage — would require a hand-rolled
  `{doc_id, term_id} → weight` relation with O(|terms| × |docs|) scan complexity
- The Seismic library (SIGIR 2024, Rust) provides state-of-the-art SPLADE retrieval if
  this changes. See docs/future-improvements.md.

**Compensation:** BM25 with camelCase/snake_case normalization on the `normalized` field
before FTS indexing. This handles the main weakness of BM25 for code (identifier tokenization).

---

## ADR-004: Provider-agnostic embeddings with fastembed as default

**Decision:** Define `EmbedProvider` trait in core; ship `FastEmbedProvider` in a separate
optional crate using fastembed-rs.

**Why:** No single embedding provider is right for all users:
- fastembed: zero external deps, code-specialized (jina-v2-base-code), offline-capable
- Ollama: free, local, good quality, requires Ollama running
- OpenAI: strong general quality, costs money, requires API key
- Voyage-code-3: best code quality benchmarks, cloud-only, costs money

Making the provider a trait means users can inject any of these — or their own — without
recompiling skelesearch-core.

**Default:** `FastEmbedProvider` with `jina-embeddings-v2-base-code` (768-dim). Zero setup,
works offline, code-specialized.

---

## ADR-005: text-splitter CodeSplitter as chunking foundation

**Decision:** Use `text-splitter` crate (benbrandt) `CodeSplitter` as the chunking algorithm
implementation, wrapped by our `LanguageConfig` trait.

**Why:** `text-splitter` implements the cAST-validated recursive merge algorithm (recurse into
oversized AST nodes, greedily merge small siblings). Writing this from scratch would replicate
existing well-tested work. Our contribution is the `LanguageConfig` trait that adds:
- Per-language chunk boundary node types (not just "all named nodes")
- Import query layer (S-expression patterns for edge extraction)
- camelCase/snake_case normalization on the normalized field

**Reference:** cAST paper (2025) shows 1.2–4.3 point retrieval gains over fixed-size chunking.
Measure chunks in non-whitespace characters, not lines or raw character count.

---

## ADR-006: Flat SQLite manifest for incremental hashing

**Decision:** Store the file hash manifest (path → mtime, size, xxHash3) in a separate
SQLite file, not in CozoDB.

**Why:** Checking N files for changes on each index run requires O(N) lookups. Doing this
through Datalog queries would incur parsing, planning, and execution overhead per lookup.
SQLite with a primary-key lookup is ~microseconds per file. The manifest and the vector
index are different workloads with different access patterns — keeping them separate is cleaner.

**Hash function:** `twox-hash` xxHash3 (31 GB/s, non-cryptographic). mtime+size checked first
as a fast pre-filter — xxHash3 computed only when metadata signals a change.

---

## ADR-007: rmcp 0.16 for MCP server

**Decision:** Use `modelcontextprotocol/rust-sdk` (rmcp crate v0.16) for the MCP server binary.

**Why:** It is the official Anthropic-maintained Rust MCP SDK. `#[tool(tool_box)]` + `schemars`
gives automatic JSON Schema generation for tool inputs. Minimal boilerplate. Stdio transport
works with Claude Code's MCP configuration out of the box.

**Note:** The community fork `4t145/rmcp` is a predecessor; use the official crate.

---

## ADR-008: Watch mode is a separate `watch` subcommand, opt-in in v1

**Decision:** `skelesearch watch <path>` is a separate CLI subcommand. There is no `--watch`
flag on `index`, and no automatic background watcher.

**Why:** The embedding step (5–50 chunks/sec depending on provider) is expensive enough that
auto-triggering on every save would be disruptive. Keeping watch as a separate subcommand
makes the "I'm now running a daemon" distinction explicit. Watch mode uses `notify` 6.x +
`notify-debouncer-full` (handles vim's rename-over-tempfile pattern). Debounce window: 1s.

---

## ADR-009: Standalone repo (not a crate in the extension crate)

**Decision:** skelesearch is a standalone repository, not a crate added to the extension crate.

**Why:** skelesearch is a product with its own versioning, packaging, and release lifecycle.
As a crate in the extension crate it would be coupled to that workspace version and harder to install
independently. Being standalone means `nix run github:you/skelesearch` works cleanly.

**Relationship to companion services:** skelesearch imports specific crates from the companion agent
service's extension crate as git dependencies (provider crates). It is not a fork — it reuses
the companion agent service's provider infrastructure without owning it.

---

## ADR-010: Individual tree-sitter grammar crates (not a bundle)

**Decision:** Add individual grammar crates (tree-sitter-rust, tree-sitter-nix, etc.) as
explicit Cargo dependencies, not a bundle crate.

**Why:** No dominant bundle crate exists in the Rust ecosystem (unlike Python's tree-sitter-languages).
Individual crates compile faster (no unused grammars), have precise version pinning, and allow
feature-flagging languages. Add grammars only for languages that have a LanguageConfig impl.
