# Repo Structure Plan: OSS Engine + Closed Service

## Decision: Mono-repo now, split at launch

Don't split repos yet. Splitting creates coordination overhead (cross-repo PRs,
version pinning, CI duplication) that kills dev velocity. Instead:

1. **Now through SOTA verification:** single repo, move fast
2. **At launch:** extract `services/` into a private repo (1 day of work)
3. **The boundary is a directory, not a repo** — clean separation by convention

## Target Structure

```
skelesearch/                          # PUBLIC repo (Apache-2.0)
├── crates/
│   ├── core/                         # Retrieval engine (schema, searcher, indexer, chunker...)
│   ├── embed-fastembed/              # Local ONNX embeddings
│   ├── embed-openai/                 # OpenAI provider
│   ├── embed-voyage/                 # Voyage provider
│   ├── rerank-api/                   # Cloud reranker client
│   ├── rerank-local/                 # Local ONNX reranker (WIP)
│   ├── mcp/                          # MCP server (local, basic) — STAYS OPEN
│   └── cli/                          # CLI binary
├── benchmarks/                       # Eval framework, adapters, cases
├── docs/                             # Technical docs, ADRs, specs
├── BENCHMARKS.md                     # Tracked results
├── README.md
├── flake.nix
└── Cargo.toml

skelesearch-cloud/                    # PRIVATE repo (proprietary)
├── services/
│   ├── api/                          # Hosted API (auth, billing, rate limiting)
│   ├── indexer-worker/               # Background indexing (GitHub webhook → index)
│   ├── dashboard/                    # Web dashboard (repo status, query logs)
│   └── landing/                      # Marketing site (Astro)
├── infra/
│   ├── fly.toml                      # Fly.io deployment
│   ├── terraform/                    # R2, DNS, Cloudflare Workers
│   └── docker/                       # Container builds
├── skelesearch = { git = "...", branch = "main" }  # Git dependency on public repo
└── Cargo.toml                        # Workspace that depends on skelesearch-core
```

## How the private repo consumes the public one

The hosted service is a Rust binary that depends on `skelesearch-core` as a
git dependency:

```toml
# skelesearch-cloud/Cargo.toml
[dependencies]
skelesearch-core = { git = "https://github.com/SecBear/skelesearch", branch = "main" }
```

This means:
- Engine improvements in the public repo immediately benefit the hosted service
- The hosted service adds auth, billing, multi-tenant isolation, webhook handling
- No code duplication — one source of truth for retrieval logic

## What goes where — the decision matrix

| Feature | Public repo | Private repo | Gate |
|---|---|---|---|
| BM25 + vector + RRF hybrid search | x | | |
| AST-aware chunking (15 languages) | x | | |
| Import graph + PageRank | x | | |
| Cross-encoder reranking | x | | |
| LLM query expansion | x | | |
| CLI (index, search, eval, watch) | x | | |
| Local MCP server (stdio) | x | | |
| Benchmarks + eval framework | x | | |
| **Hosted MCP endpoint** | | x | API key |
| **GitHub webhook auto-reindex** | | x | Pro tier |
| **Multi-repo management** | | x | Pro tier |
| **Team/org shared indexes** | | x | Team tier |
| **Usage dashboard + query logs** | | x | Pro tier |
| **Rate limiting + auth** | | x | All tiers |
| **Priority indexing queue** | | x | Team tier |
| **Landing page + docs site** | | x | |

## The pro MCP server pattern

The local MCP server (`crates/mcp/`) stays open and works standalone.
The hosted service exposes the same MCP protocol but over HTTP:

```json
// Claude Code config — local (free, open source)
{
  "mcpServers": {
    "skelesearch": {
      "command": "skelesearch",
      "args": ["mcp"]
    }
  }
}

// Claude Code config — hosted (paid, cloud)
{
  "mcpServers": {
    "skelesearch": {
      "url": "https://api.skelesearch.dev/mcp",
      "headers": {
        "Authorization": "Bearer sk-..."
      }
    }
  }
}
```

The upgrade path is changing one config block. No code change, no reinstall.

## What to build NOW (pre-launch)

### Phase 1: Verify SOTA (this week)
- [x] CoIR CoSQA run (0.442 nDCG@10)
- [ ] CoIR full 10-task run
- [ ] ContextBench pilot
- [ ] ContextBench verified (500 instances)
- [ ] Update BENCHMARKS.md with all results

### Phase 2: Launch prep (1-2 weeks)
- [ ] Submit CoIR results to MTEB leaderboard
- [ ] Clean up README for public consumption
- [ ] Record 60-second demo: Claude Code + skelesearch MCP
- [ ] Landing page (Astro on Cloudflare Pages)
- [ ] Stripe checkout (Pro tier)
- [ ] Waitlist for hosted service

### Phase 3: Hosted service MVP (2-4 weeks post-launch)
- [ ] Create private skelesearch-cloud repo
- [ ] API server: auth (API keys), rate limiting, Fly.io
- [ ] Indexer worker: accept GitHub webhook, clone, index, store
- [ ] R2 storage for index artifacts
- [ ] Free tier endpoint (1 repo, rate limited)
- [ ] Pro billing integration

## What NOT to build before launch

- SSO / SAML
- Team management UI
- VS Code extension
- Multiple regions
- Enterprise contracts
- Admin dashboard beyond basic metrics

These are post-traction features. Ship with API keys + Stripe + one region.

## License structure

```
skelesearch (public)     → Apache-2.0 OR MIT (dual license, standard Rust pattern)
skelesearch-cloud        → Proprietary (all rights reserved)
```

The dual Apache-2.0/MIT is already in place (LICENSE-APACHE, LICENSE-MIT exist).
No change needed on the public side.
