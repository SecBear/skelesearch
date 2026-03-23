# Eval Failure Analysis: zod & httpx

Generated: 2026-03-21

## Summary

Both zod (63.6% MRR) and httpx (65.2% MRR) are the weakest repos in the eval.
This document identifies why specific cases likely fail and what was fixed.

---

## Zod

### Root Cause 1: v3/v4 File Overlap (HIGH IMPACT)

The zod repo contains BOTH `packages/zod/src/v3/` and `packages/zod/src/v4/` in the same
index. Several files exist in both:

| v3 file | v4 file | Competes on |
|---------|---------|-------------|
| `v3/ZodError.ts` | `v4/core/errors.ts` | "ZodError" keyword, error collection queries |
| `v3/errors.ts` | `v4/core/errors.ts` | error-related terms |
| `v3/standard-schema.ts` | `v4/core/standard-schema.ts` | "Standard Schema" terms |
| `v3/types.ts` (5138 lines) | `v4/core/schemas.ts` (4556 lines) | schema type definitions |

**Critical detail for standard-schema**: The v3 type is named `StandardSchemaV1` which
contains "Schema" — strong BM25 match for "Standard Schema protocol". The v4 type was
renamed to `StandardTypedV1`, losing the BM25 keyword advantage. This means BM25 will
rank v3/standard-schema.ts ABOVE v4/core/standard-schema.ts for the standard schema query.
Semantic search must resolve this. **Cases at risk**: [16], [20], [26].

### Root Cause 2: Wrong/Incomplete expected_files (FIXED)

Three cases had incorrect or incomplete expected_files:

**Case 30 — "string-bool coercion"** (FIXED: added `core/api.ts`)
- `classic/schemas.ts` is a 3-line wrapper calling `core._stringbool()`
- The actual conversion logic (truthy/falsy arrays, case handling, Set construction) is in
  `core/api.ts` lines ~1713-1737. This was missing from expected_files.

**Case 31 — "hash IDs for deduplication"** (FIXED: `util.ts` → `registries.ts`)
- Premise is wrong: zod does NOT use deterministic hash IDs for schema deduplication.
- Actual mechanism: `$ZodRegistry` uses WeakMap for schema→metadata and a string-id `_idmap`.
- `util.ts` has `HashAlgorithm` types but for validating hash string FORMATS, not schema identity.
- `registries.ts` is the correct answer (WeakMap, _idmap, singleton globalRegistry pattern).

**Case 32 — "doc utility documentation metadata"** (FIXED: `doc.ts` → `registries.ts` + `api.ts`)
- `doc.ts` exports `class Doc` which is a JIT code-generation builder — used by `schemas.ts`
  to `new Function()` compile object validation at runtime. Nothing to do with metadata.
- Actual documentation metadata: `globalRegistry.add(schema, {title, description, ...})`
  via `registries.ts`; `.meta()` check in `api.ts` line 1684.

### Root Cause 3: Architecture Cases with Small Files

**Case 37 — "internal schema definition layer from classic public-facing API"**
- `core/core.ts` is only 138 lines. The phrase "internal schema definition layer" doesn't
  appear literally in the file. The file defines `$constructor` factory pattern.
- BM25 won't find it. Semantic search must understand that `$constructor` + `_zod.traits`
  IS the internal definition layer. This is hard.

### Root Cause 4: Cross-File Cases Need Graph Augmentation

Cases in `cross_file` and `architecture` categories often expect 2-3 files. If graph
augmentation is not reaching the secondary files (e.g., `core/api.ts` when starting from
`classic/schemas.ts`), these cases score 0.5 instead of 1.0 in MRR.

Affected cases: [1], [5], [10], [13], [22], [25], [28], [37].

### Root Cause 5: Barrel Penalty on Index Files

`classic/index.ts` and `core/index.ts` are barrel files (all re-exports) and receive 0.4x
penalty. Case 22 expects both of these. Strong semantic signal from "initialised relative to
core layer on import" should overcome the penalty since the query is explicitly about entry
points.

### Category Failure Rate Predictions

| Category | # Cases | Expected Hard Cases | Reason |
|----------|---------|---------------------|--------|
| `implementation` | 13 | 2-3 | Large files, BM25 can find them |
| `cross_file` | 5 | 2-3 | Requires graph aug for secondary files |
| `error_handling` | 5 | 2-3 | v3 file competition |
| `config_init` | 4 | 1-2 | Barrel penalty |
| `vocabulary_gap` | 4 | 2-3 | Non-literal terms + v3 competition |
| `symbol_lookup` | 5 | 0-1 | Clean symbol names, direct matches |
| `architecture` | 4 | 2-3 | Small files, multi-file cases |
| `utility_helper` | 4 | 1-2 | One case fixed (doc/hash) |

---

## HTTPX

### Root Cause 1: `__init__.py` Barrel Dominance

`httpx/__init__.py` has 80 symbols in `__all__` and 12 wildcard `from X import *` lines.
Every keyword from every module appears in this file, giving it artificially high BM25.
The 0.4x barrel penalty reduces this but it may still compete with implementation files
for queries that name specific classes/functions.

**Good news**: The case asking about the public namespace (case 17) explicitly expects
`__init__.py`. That query's semantic embedding should survive the 0.4x penalty.

**Possible remaining issue**: For implementation queries about, say, `DigestAuth`, `__init__.py`
re-exports it from `_auth.py`. After 0.4x, `_auth.py` should win because it has 250+ lines
of actual implementation vs `__init__.py`'s one-line re-export. Monitor `implementation`
category cases if MRR is still low.

### Root Cause 2: Test File Competition

`tests/` directory is large and contains tests for auth, transport, client, etc. These
test files match keywords from implementation queries. The 0.3x test penalty handles this
but deserves monitoring. Case 21 ("mock transport for intercepting requests in tests") is
especially at risk because "tests" appears in the query itself.

**Note**: `httpx/_transports/mock.py` is a PRODUCTION file, not in tests/. It should rank
above test files even without penalty since its entire content is about MockTransport.

### Root Cause 3: Implementation-in-One-File-Reexported-Through-Another

Several cases involve functions defined in one file but accessible via another:
- `QueryParams` defined in `_urls.py`, accessed via `__init__.py`
- `BasicAuth`/`DigestAuth` defined in `_auth.py`, accessed via `__init__.py`

With the barrel penalty, direct implementation files should rank higher. These cases
should work correctly already.

### Category Failure Rate Predictions

| Category | # Cases | Expected Hard Cases | Reason |
|----------|---------|---------------------|--------|
| `implementation` | 11 | 2-3 | __init__.py competition, barrel penalty |
| `cross_file` | 6 | 1-2 | Multi-file, graph augmentation |
| `error_handling` | 4 | 0-1 | Clear exception hierarchy |
| `config_init` | 4 | 0-1 | Clear config files |
| `vocabulary_gap` | 2 | 0-1 | IDNA, transport — clear files |
| `symbol_lookup` | 5 | 1-2 | __init__.py barrel penalty |
| `architecture` | 4 | 0-1 | Clear architecture files |
| `utility_helper` | 4 | 0-1 | Clear utility files |

---

## Scoring Changes: Decision

After analysis, **no scoring changes were made**. Rationale:

1. **Barrel penalty (0.4x)**: The cases where barrel files ARE the expected answer
   (`__init__.py` namespace query, `classic/index.ts`+`core/index.ts` init query) have
   strong semantic signal that should survive 0.4x. Reducing to 0.2x for "pure barrels"
   would risk breaking exactly these cases without empirical justification.

2. **v3/v4 boost**: Adding a penalty for v3/ paths or a boost for v4/ paths would be
   zod-specific overfitting. The correct solution is to ensure the LLM query expander
   adds "v4" to queries when relevant, or to selectively exclude v3/ files from the index.

3. **Small file scoring**: `core/core.ts` (138 lines) is genuinely hard to surface via BM25.
   No scoring change helps — the issue is semantic: does the embedder understand that
   `$constructor` factory pattern IS the "internal schema definition layer"?

The annotation fixes are the high-value interventions: correcting wrong expected_files
changes the ground truth, which affects MRR measurement directly.
