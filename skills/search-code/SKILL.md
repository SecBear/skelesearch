# search-code

## Use when
- Locating function definitions, struct declarations, or symbol usages across the codebase
- Finding files relevant to a feature, bug, or refactor before reading them
- Confirming whether a pattern, API, or identifier exists anywhere in the project

## How it works
`skelesearch` indexes source files with tree-sitter AST chunking and FAISS-backed semantic search.
Query results are ranked by vector similarity against the embedded query string.

## CANDIDATES to verify
Results are **candidates**, not assertions. The search returns the most likely relevant chunks;
always read the full context before concluding the code does or does not do something.
False positives are possible when symbol names are overloaded or similar across files.

## Invocation
```
skelesearch search "<query>" [--limit N] [--json]
```
Use `--json` when feeding results into downstream processing.
