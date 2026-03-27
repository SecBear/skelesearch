# skelesearch quickstart

## 1. Build

```bash
# Development build (fast compile)
cargo build

# Install into PATH
cargo install --path crates/cli
```

For a release build with the RocksDB backend (slower first compile, better write
throughput at scale):

```bash
# macOS may need: export CXXFLAGS="-std=c++20"
cargo install --path crates/cli --features skelesearch-core/storage-rocksdb
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

The response includes `indexed_files`, `total_chunks`, `last_indexed`,
`estimated_stale`, and `freshness_state`.

- `freshness_state` is one of `fresh`, `stale`, `refreshing`, or `unknown`
- `watching` is separate from freshness and only tells you whether the watch loop is alive
- `estimated_stale` is a best-effort manifest-based count, not a hard guarantee

When status is `stale`, re-run `skelesearch index .` (or let startup/watch remediation catch up).

## 3. CLI search

```bash
skelesearch search "retry logic with exponential backoff" --top-k 10 --json
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
| `get_index_status` | Check indexing progress, freshness state, stale estimate, and watcher state |

## 5. Re-indexing

Re-run `skelesearch index .` (or call `index` via MCP) after:

- A fresh clone on a new machine
- Large merges or rebases
- Any time `get_index_status` reports `stale`, `unknown`, or indexing errors

The `hooks/post-edit-reindex` hook re-indexes automatically in the background after
each file edit when skelesearch is wired via the Claude Code plugin manifest.

## 6. Index location

`.skelesearch/` at the project root contains the active index state. In current
builds that may be either:

- legacy root files: `manifest.db` and `index.db`
- generation-backed state: `active-generation` plus `generations/<id>/manifest.db`
  and `generations/<id>/index.db`

These files are machine-local. Do not commit them.
