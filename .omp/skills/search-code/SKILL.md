# search-code

## Use when
- Locating function definitions, struct declarations, or symbol usages across the codebase
- Finding files relevant to a feature, bug, or refactor before reading them
- Confirming whether a pattern, API, or identifier exists anywhere in the project
- Navigating an unfamiliar codebase before making a change

## How it works
`skelesearch` indexes source files with tree-sitter AST chunking and CozoDB HNSW-indexed
semantic search. Query results are ranked by hybrid BM25+dense similarity against the
embedded query string.

## CANDIDATES to verify
Results are **candidates**, not assertions. The search returns the most likely relevant
chunks; always read the full context before concluding the code does or does not do
something. False positives are possible when symbol names are overloaded or similar
across files.

## CLI invocation
```
skelesearch search "<query>" [--limit N] [--json]
```
Use `--json` when feeding results into downstream processing.

## MCP tools

When skelesearch is running as an MCP server, the following tools are available:

| Tool | When to use |
|------|------------|
| `smart_search` | **Start here for ambiguous queries.** Classifies the query and selects the best retrieval strategy automatically. |
| `search_code` | Explicit hybrid BM25+dense search. Accepts `include_graph: true` to expand results with import-graph neighbours. |
| `find_symbol` | Look up a specific function, struct, or type by exact or partial name. Faster than `search_code` for known identifiers. |
| `get_file_context` | Retrieve all indexed chunks and import edges for a specific file path. |
| `index_codebase` | Trigger full indexing of a directory from within a Claude session. |
| `index_status` | Check whether the index exists and how current it is. |

### If the project is not indexed yet
Call `index_codebase` with the project root path. Indexing must complete before any
search or context tool will return results. Check progress with `index_status`.

### Strategy guidance
- **Ambiguous or conceptual queries** (e.g., "how is authentication handled"): use `smart_search`.
- **Known symbol name** (e.g., "where is `parse_config` defined"): use `find_symbol`.
- **Broad feature exploration** (e.g., "everything touching the retry logic"): use `search_code`
  with `include_graph: true` to pull in transitive import neighbours.
- **File-level review** (e.g., "what does this file import and what imports it"): use
  `get_file_context`.
