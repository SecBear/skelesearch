# skelesearch quickstart

## 1. Build

```bash
# Development build (SQLite backend, fast compile)
cargo build --features storage-sqlite

# Install into PATH
cargo install --path crates/cli --features storage-sqlite
```

For a release build with the RocksDB backend (slower first compile, better write
throughput at scale):

```bash
# macOS may need: export CXXFLAGS="-std=c++20"
cargo install --path crates/cli --features storage-rocksdb
```

## 2. First-run indexing

From your project root:

```bash
skelesearch index .
```

The index is stored in `.skelesearch/` at the project root. Add `.skelesearch/` to
`.gitignore` — the index is machine-local and rebuilds quickly.

Check that indexing completed:

```bash
skelesearch status --json
```

The response includes `indexed_files`, `total_chunks`, `last_indexed`, and
`estimated_stale`. When `estimated_stale` is high, re-run `skelesearch index .`.

## 3. CLI search

```bash
skelesearch search "retry logic with exponential backoff" --limit 10 --json
```

Results are **candidates** — they are ranked by semantic similarity, not proven matches.
Always read the surrounding code before drawing conclusions.

## 4. MCP server setup (Claude Code)

Start the MCP server:

```bash
skelesearch-mcp   # binary built from crates/mcp
```

Add it to your Claude Code MCP config (`.claude/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "skelesearch": {
      "command": "skelesearch-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

If `skelesearch-mcp` is not in PATH, use the full path to the binary instead.

Once connected, Claude Code can call these tools:

| Tool | Purpose |
|------|---------|
| `search_code` | Primary hybrid code search with optional graph expansion |
| `find_symbol` | Look up definitions by exact symbol name |
| `get_symbol_info` | One-call context bundle for a known symbol |
| `find_dependents` | Find files that import or otherwise depend on a target file |
| `find_tests` | Find tests covering a source file |
| `get_repo_map` | Fast structural overview of the indexed repo |
| `index` | Trigger background indexing from within a session |
| `get_index_status` | Check indexing progress and freshness |

## 5. Re-indexing

Re-run `skelesearch index .` (or call `index` via MCP) after:

- A fresh clone on a new machine
- Large merges or rebases
- Any time `get_index_status` shows stale files or indexing errors

The `hooks/post-edit-reindex` hook re-indexes automatically in the background after
each file edit when skelesearch is wired via the Claude Code plugin manifest.

## 6. Index location

`.skelesearch/` at the project root contains:

- `manifest.db` — SQLite manifest tracking indexed files and chunk metadata
- `index.db` (or RocksDB-backed equivalent) — the embedded vector and graph database

These files are machine-local. Do not commit them.
