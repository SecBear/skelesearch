---
name: competitive-research-roadmap
description: >
  Generate comprehensive competitive landscape analysis and feature roadmap for
  any software project. Use when a project needs competitive positioning, feature
  gap analysis, benchmarking strategy, or a prioritized improvement roadmap.
  Triggers: "competitive analysis", "how do we compare", "what are competitors doing",
  "feature roadmap", "beat the competition", "market positioning", "landscape analysis",
  "what's missing", "gap analysis", "prioritize features".
---

# Competitive Research & Roadmap Generation

Produce two artifacts for any software project:
1. **Competitive Landscape** — structured analysis of all competitors with feature matrices
2. **Feature Roadmap** — phased, prioritized spec to achieve market leadership

## Process

### Phase 1: Project Audit (5 min)

Understand what the project does and what it already has before researching anything.

1. Read the project's README, AGENTS.md, main design docs
2. List every feature the project currently has (be specific — not "search" but "hybrid BM25+vector with RRF fusion")
3. Identify the project's core value proposition in one sentence
4. Identify the target user persona (developer? agent? enterprise?)

### Phase 2: Competitive Discovery (parallel, 8-10 min)

Dispatch a `researcher` agent with these exact research questions:

```
1. DIRECT COMPETITORS
Find ALL tools/products that solve the same problem. Include:
- Open source projects (GitHub stars, language, license, activity)
- Commercial products (pricing, published claims)
- Features built into larger platforms
For each: name, URL, approach, retrieval method, differentiator, maturity.

2. BENCHMARKS
What benchmarks exist for measuring quality in this domain?
For each: what it measures, metrics, who uses it, relevance.

3. COMPETITOR CLAIMS
What specific numbers do competitors publish?
Search blog posts, READMEs, papers. Be specific.

4. BEST PRACTICES
How do the market leaders solve this problem?
What approaches exist? What's the consensus? What's controversial?
```

### Phase 3: Novel Techniques Research (parallel, 6-8 min)

Dispatch a second `researcher` agent for cutting-edge approaches:

```
For each technique relevant to the project's domain:
- What it is (1-2 sentences)
- State of the art (who, what results)
- Implementation complexity (low/medium/high)
- Expected impact
- Open-source availability (libraries, models, papers with code)
- Recommendation (implement now / later / skip)

Also search for:
- Techniques NOT in the known list that could be game-changers
- What top academic groups published in 2025-2026
- Emerging standards or protocols
```

### Phase 4: Synthesize Competitive Landscape (10 min)

Write `docs/competitive-landscape.md` with these sections:

```markdown
# Competitive Landscape — [Date]

## Direct Open-Source Competitors
| Tool | Stars | Language | Key Features | MCP/API | License |

## Commercial / Built-in
| Tool | Approach | Published Claims | Key Insight |

## Benchmarks
| Benchmark | What | Metrics | Relevance |

## Top Models/Approaches
| Model/Approach | Provider | Key Stats | Verdict |

## Positioning
- Unique combination the project offers
- Market gap being targeted
```

Key principles:
- Feature matrices over prose — readers compare columns, not paragraphs
- Include specific numbers (stars, pricing, published metrics) not vague qualifiers
- Note dead/stalled projects explicitly — they create market gaps
- Identify "closest competitor" and "biggest threat" separately

### Phase 5: Synthesize Roadmap (15 min)

Write `docs/v2-roadmap.md` (or `vN-roadmap.md`) with these sections:

```markdown
# [Project] vN Roadmap — Beat Every Competitor

## Current State
What the project has today (specific features, not vague categories).

## Phase 1: Production Polish (quick wins, 1-2 days)
Features that close dogfooding gaps or fix bugs that embarrass the project.

## Phase 2: Competitive Parity (1 week)
Features that match what the best competitors already have.

## Phase 3: Competitive Advantage (2-3 weeks)
Features that NO competitor has — the moat.

## Phase 4: Measurement (3-5 days)
Eval framework to prove quality with numbers.

## Phase 5: Advanced (2-4 weeks)
Market leadership features.

## Phase 6: Ecosystem (ongoing)
Distribution, integrations, documentation.

## Priority Matrix
| Feature | Effort | Impact | Competitors Have It | Priority |

## What Makes This "Can't Live Without"
The thesis: why this project wins.

## Technical Decisions Log
| Decision | Choice | Rationale |
```

Key principles:
- Each phase has a clear time estimate and a one-line theme
- Features are concrete (file paths, API changes, wire formats) not vague
- Priority matrix uses P0/P1/P2/P3 with effort and impact columns
- "What Makes This Can't Live Without" is the pitch — one paragraph, specific

### Phase 6: Commit

Commit both files together with a descriptive message referencing the research sources.

## Quality Checklist

Before delivering, verify:
- [ ] Every competitor has a URL, not just a name
- [ ] Feature matrices have at least 6 columns for meaningful comparison
- [ ] Published benchmark numbers are cited with source
- [ ] Roadmap phases have time estimates
- [ ] Priority matrix covers every feature in the roadmap
- [ ] "Can't live without" section is specific, not generic
- [ ] Dead/stalled projects are explicitly marked
- [ ] At least one novel technique is recommended for "implement now"
- [ ] The roadmap identifies at least one feature NO competitor has

## Anti-patterns

- Writing prose where a table would be clearer
- Listing competitors without feature comparison
- Roadmap without time estimates or priorities
- Claiming "best" without benchmark data
- Recommending techniques without implementation complexity assessment
- Ignoring dead projects (they create market gaps worth noting)
- Separate "research" and "action" documents (combine into one roadmap)
