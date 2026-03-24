# skelesearch integrations

Practical guide for wiring skelesearch into different agentic systems.
All configurations assume `skelesearch-mcp` and `skelesearch` are in `$PATH`
(see [quickstart.md](quickstart.md) for build and install steps).

---

## 1. Claude Code (MCP stdio)

Add to `.claude/mcp.json` at the project root (or the global Claude Code MCP
config):

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

Once connected, Claude Code can call these tools:

| Tool | Purpose |
|------|---------|
| `search_code` | Primary hybrid code search with graph expansion, diversity, and token budget controls |
| `find_symbol` | Look up definitions by name (and optional `kind`) |
| `get_symbol_info` | One-call symbol context bundle for a known symbol |
| `find_dependents` | Find files that import or depend on a file |
| `find_tests` | Discover tests covering a file |
| `get_repo_map` | Fast structural overview of the indexed repo |
| `index` | Trigger background indexing from within a session |
| `get_index_status` | Check whether indexing is current or still running |

**Tip:** The `diversity` parameter defaults to `0.3` for MCP calls, which
re-ranks results with Maximal Marginal Relevance to reduce redundancy. Lower
it toward `0.0` if you want strict similarity order; raise it toward `1.0`
for maximum coverage across different parts of the codebase.

See [quickstart.md § 4](quickstart.md) for the full Claude Code setup walkthrough.

---

## 2. Codex (OpenAI)

Codex discovers MCP tools automatically via the `list_tools` protocol message.
Add skelesearch to the Codex MCP config — typically `~/.codex/config.json`
for user-level or a project-level equivalent (check your Codex version docs,
as the config path has varied across releases):

```json
{
  "mcpServers": {
    "skelesearch": {
      "command": "skelesearch-mcp",
      "args": []
    }
  }
}
```

Recommended workflow:

1. Run `skelesearch index .` once before starting a Codex session.
2. Codex will pick up all six tools via `list_tools` on connect.
3. Use `search_code` throughout the session; call
   `index` to re-index after large edits.

> **Note:** Codex config format and MCP support details have changed across
> versions. If the above doesn't work, check the [Codex documentation](https://platform.openai.com/docs/guides/codex)
> for the current MCP configuration schema.

## 3. Custom MCP host (stdio)



For any MCP host that launches tools as local subprocesses, configure
`skelesearch-mcp` as a stdio server and pass environment variables through the
host's normal MCP config mechanism.



Example install:

```bash

cargo install --path crates/mcp --root ~/.local --force --features storage-rocksdb

# or use the helper script

./scripts/install-mcp.sh

```



Example MCP entry:

```json

{

  "mcpServers": {

    "skelesearch": {

      "command": "skelesearch-mcp",

      "env": {

        "VOYAGE_API_KEY": "<set-me>",

        "RUST_LOG": "skelesearch=info"

      }

    }

  }

}

```



Workflow:

1. Install or rebuild the MCP binary

2. Update your host's MCP config

3. Restart the host

4. skelesearch will auto-index on startup when needed (unless `SKELESEARCH_NO_AUTO_INDEX` is set)



Tool recommendations:

- **`search_code`** — primary default for semantic code retrieval.

- **`find_symbol`** — precise definition lookup. Pass `kind` to narrow
  to a specific symbol type (e.g. `"struct"`, `"function"`, `"trait"`).

- **`get_symbol_info`** — best one-call context bundle for a known symbol.

- **`get_repo_map`** — fastest structural overview when you need repo shape before querying.



---
---

## 4. HTTP transport (generic)

> **New in v1.2.** For systems that cannot spawn subprocesses directly —
> VS Code extensions, remote agents, web-based tools — skelesearch-mcp will
> expose a Streamable HTTP transport:

```bash
skelesearch-mcp --http 127.0.0.1:3000
```

- JSON-RPC requests arrive via HTTP POST.
- Streaming responses (where applicable) use SSE.
- Compatible with any MCP client that supports the Streamable HTTP transport.

This transport is available in current builds. Use stdio transport only if your
client lacks Streamable HTTP support.

---

## 5. Custom integration via CLI

For systems without MCP support, the `skelesearch` CLI produces JSON output
suitable for programmatic consumption.

```bash
# Index the current project
skelesearch index .

# Semantic search — JSON array of ranked chunks
skelesearch search "error handling" --top-k 10 --json --diversity 0.3

# Symbol lookup — narrow by kind if needed
skelesearch symbol "MyStruct" --kind struct

# Regex grep across indexed files
skelesearch grep "TODO|FIXME" --json

# Index health check
skelesearch status --json
```

`--diversity` defaults to `0.0` in the CLI (strict similarity order, backward
compatible). Pass `0.3` to get the same MMR re-ranking behavior as the MCP
server default.

Parse the JSON output with `jq`, Python, or any JSON library. The schema is
stable within a major version.

---

## 6. Post-edit reindexing hook

The `hooks/post-edit-reindex` script keeps the index current automatically.

**Wire it into your editor or CI:**

- **Claude Code plugin:** configure your plugin manifest to run `hooks/post-edit-reindex`
  via a post-file-edit hook if you want automatic background reindexing.
- **Other editors:** Point your editor's on-save hook at
  `hooks/post-edit-reindex` in the project root. The script spawns
  `skelesearch index .` in the background and exits immediately; it will not
  block your save.
- **CI pipeline:** Run `hooks/post-edit-reindex` as a post-checkout or
  post-merge step to ensure the index reflects the latest HEAD.

**Behavior:**

- If a `skelesearch watch` sentinel is active, the hook detects it via the
  lock file and exits 0 without spawning a duplicate indexer.
- Concurrent saves are safe: the script uses an exclusive lock
  (`.skelesearch/.skelesearch.lock`) so at most one indexer runs at a time.
- The hook only re-embeds files whose content has changed since the last run
  (embedding cache), so subsequent runs are fast.

---

## Tips

**First run is the slowest.** The initial `skelesearch index .` downloads the
embedding model and processes every file. Subsequent runs skip unchanged files
via the embedding cache and complete in seconds for typical codebases.

**Keep `.skelesearch/` out of version control.** The index is machine-local
and rebuilds quickly. Add it to `.gitignore`:

```
.skelesearch/
```

**Exclude noise in large monorepos.** Add patterns to
`.skelesearch.toml` to skip generated code, vendored dependencies, or
large binary directories:

```toml
[index]
exclude = ["vendor/", "node_modules/", "target/", "dist/"]
```

**`diversity` quick reference:**

| Value | Effect |
|-------|--------|
| `0.0` | Strict cosine similarity order (CLI default) |
| `0.3` | Light MMR re-ranking — less redundancy (MCP default) |
| `1.0` | Maximum diversity — broad coverage, lower per-result precision |

**Choosing a tool:**

- Semantic code search / unknown intent → `search_code`
- Definition lookup → `find_symbol`
- One-shot symbol context → `get_symbol_info`
- Repo structure / import overview → `get_repo_map`
- Dependency impact → `find_dependents`
- Test discovery → `find_tests`
- Pattern matching → `grep` (CLI)
