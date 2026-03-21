#!/usr/bin/env python3
"""CoIR (Code Information Retrieval) benchmark adapter for skelesearch.

Supports two evaluation modes:

  embed-only  Wraps skelesearch's embedding provider as a BEIR-compatible
              encoder.  Measures pure embedding quality — scores are directly
              comparable to published Voyage/OpenAI/Jina results on MTEB/CoIR.

  hybrid      Wraps skelesearch's full retrieval pipeline (index + BM25 +
              vector + RRF + optional reranker).  The corpus is written to a
              temp directory and indexed, then each query is run through the
              binary.  This measures skelesearch's end-to-end retrieval quality
              over function-level code snippets.

Primary metric: nDCG@10 (higher is better).

Usage:
  # Embed-only — no binary needed, fast smoke test
  uv run --with coir-eval --with fastembed --with numpy \\
    python3 benchmarks/scripts/coir-eval.py \\
    --mode embed-only --provider fastembed \\
    --tasks cosqa,apps \\
    --output benchmarks/runs/coir-fastembed.json

  # Full embed-only run (all tasks)
  uv run --with coir-eval --with fastembed --with numpy \\
    python3 benchmarks/scripts/coir-eval.py \\
    --mode embed-only --provider fastembed \\
    --tasks all \\
    --output benchmarks/runs/coir-fastembed-full.json

  # Hybrid mode — requires a built skelesearch binary
  uv run --with coir-eval --with fastembed --with numpy \\
    python3 benchmarks/scripts/coir-eval.py \\
    --mode hybrid \\
    --binary ./target/release/skelesearch \\
    --provider fastembed \\
    --tasks cosqa \\
    --output benchmarks/runs/coir-hybrid.json

  # List available task names
  python3 benchmarks/scripts/coir-eval.py --list-tasks
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model used by skelesearch's default fastembed provider.
# Must match the Rust binary default: jina-embeddings-v2-base-code (768-dim).
FASTEMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"

# Exact task names accepted by coir.data_loader.load_data_from_hf.
# These map to HuggingFace datasets under the CoIR-Retrieval org.
STANDALONE_TASKS: List[str] = [
    "cosqa",
    "apps",
    "synthetic-text2sql",
    "codetrans-dl",
    "codetrans-contest",
    "codefeedback-st",
    "codefeedback-mt",
    "stackoverflow-qa",
]

# Group aliases expanded by coir.data_loader.get_tasks into per-language sub-tasks.
GROUP_TASKS: List[str] = [
    "codesearchnet",       # → CodeSearchNet-{go,java,javascript,ruby,python,php}
    "codesearchnet-ccr",   # → CodeSearchNet-ccr-{go,java,javascript,ruby,python,php}
]

ALL_TASK_NAMES: List[str] = STANDALONE_TASKS + GROUP_TASKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embed-only model — BEIR-compatible encoder
# ---------------------------------------------------------------------------

class EmbedOnlyModel:
    """Encoder backed by skelesearch's configured embedding provider.

    encode_queries / encode_corpus return float32 np.ndarray of shape (N, D).
    DRES (DenseRetrievalExactSearch) converts numpy arrays to tensors via its
    internal cos_sim, so returning numpy is fine and avoids an extra copy.

    Corpus items from CoIR have 'title' and 'text' keys; we concatenate them
    to form a single document string (same strategy as most BEIR baselines).
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        if provider == "fastembed":
            from fastembed import TextEmbedding  # type: ignore[import-untyped]
            logger.info("Loading fastembed model: %s", FASTEMBED_MODEL)
            self._fastembed = TextEmbedding(FASTEMBED_MODEL)
        elif provider == "voyage":
            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                sys.exit("ERROR: VOYAGE_API_KEY environment variable is not set.")
            import voyageai  # type: ignore[import-untyped]
            self._voyage = voyageai.Client(api_key=api_key)
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                sys.exit("ERROR: OPENAI_API_KEY environment variable is not set.")
            from openai import OpenAI  # type: ignore[import-untyped]
            self._openai = OpenAI(api_key=api_key)
        else:
            sys.exit(
                f"ERROR: Unknown provider '{provider}'. Choices: fastembed, voyage, openai"
            )

    # --- Per-provider embedding helpers ---

    def _embed_fastembed(self, texts: List[str], batch_size: int) -> np.ndarray:
        # fastembed.embed() is a generator; collect into a list first.
        embeddings = list(self._fastembed.embed(texts, batch_size=batch_size))
        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"fastembed returned unexpected shape {arr.shape}; expected (N, D)"
            )
        return arr

    def _embed_voyage(
        self, texts: List[str], batch_size: int, input_type: str
    ) -> np.ndarray:
        # Voyage enforces a max of 128 texts and 320k tokens per request.
        VOYAGE_BATCH = min(batch_size, 128)
        all_embs: List[List[float]] = []
        for i in range(0, len(texts), VOYAGE_BATCH):
            batch = texts[i : i + VOYAGE_BATCH]
            result = self._voyage.embed(batch, model="voyage-code-2", input_type=input_type)
            all_embs.extend(result.embeddings)
        arr = np.array(all_embs, dtype=np.float32)
        return _l2_normalize(arr)

    def _embed_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        # OpenAI allows up to 2048 inputs per request.
        OAI_BATCH = min(batch_size, 2048)
        all_embs: List[List[float]] = []
        for i in range(0, len(texts), OAI_BATCH):
            batch = texts[i : i + OAI_BATCH]
            result = self._openai.embeddings.create(
                input=batch, model="text-embedding-3-small"
            )
            all_embs.extend([r.embedding for r in result.data])
        arr = np.array(all_embs, dtype=np.float32)
        return _l2_normalize(arr)

    # --- BEIR interface ---

    def encode_queries(
        self, queries: List[str], batch_size: int = 32, **kwargs
    ) -> np.ndarray:
        logger.info("Encoding %d queries [%s]...", len(queries), self.provider)
        if self.provider == "fastembed":
            return self._embed_fastembed(queries, batch_size)
        if self.provider == "voyage":
            return self._embed_voyage(queries, batch_size, input_type="query")
        # openai
        return self._embed_openai(queries, batch_size)

    def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int = 32, **kwargs
    ) -> np.ndarray:
        """Concatenate title + text then embed.

        CoIR corpus items carry 'title' and 'text' keys.  Concatenating them
        with a newline matches the convention used by most published baselines.
        """
        logger.info("Encoding %d corpus docs [%s]...", len(corpus), self.provider)
        texts = [
            (doc.get("title") or "") + "\n" + (doc.get("text") or "")
            for doc in corpus
        ]
        if self.provider == "fastembed":
            return self._embed_fastembed(texts, batch_size)
        if self.provider == "voyage":
            return self._embed_voyage(texts, batch_size, input_type="document")
        # openai
        return self._embed_openai(texts, batch_size)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Divide each row by its L2 norm; leave zero-vectors untouched."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


# ---------------------------------------------------------------------------
# Hybrid search backend — full skelesearch pipeline
# ---------------------------------------------------------------------------

class HybridSearch:
    """Runs skelesearch's full retrieval pipeline as a BEIR search backend.

    Protocol per search() call:
      1. Write every corpus document as a .txt file in a temp directory.
      2. Index that directory with `skelesearch index`.
      3. Run `skelesearch search <query> --json` for each query.
      4. Map file_path results back to corpus doc IDs and return scores.

    This measures BM25 + dense vector + RRF + optional reranker over
    function-level code snippets, which is skelesearch's primary use case.

    Limitations:
    - Large corpora (CodeSearchNet ~1M docs) may be slow to index.
    - The temp directory is deleted after each task, so there is no cross-task
      index reuse.  Use --tasks with a single task for faster iteration.
    """

    # Characters unsafe in filenames across macOS/Linux/Windows.
    _UNSAFE_RE = re.compile(r"[^\w\-]")

    def __init__(self, binary: str, provider: str, top_k: int = 100) -> None:
        self.binary = str(Path(binary).resolve())
        self.provider = provider
        self.top_k = top_k
        if not Path(self.binary).exists():
            sys.exit(f"ERROR: skelesearch binary not found: '{self.binary}'")

    @classmethod
    def _safe_stem(cls, doc_id: str, max_len: int = 200) -> str:
        """Convert a doc_id to a filesystem-safe filename stem."""
        return cls._UNSAFE_RE.sub("_", doc_id)[:max_len]

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        **kwargs,  # absorbs score_function and other BEIR kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Index corpus + search all queries; return {qid: {doc_id: score}}."""
        effective_top_k = max(top_k, self.top_k)

        with tempfile.TemporaryDirectory(prefix="skelesearch_coir_") as tmpdir:
            tmppath = Path(tmpdir)

            # Step 1: Write corpus.
            # Build a collision-free stem → doc_id map.
            logger.info("Writing %d corpus documents to temp dir...", len(corpus))
            stem_to_docid: Dict[str, str] = {}
            for doc_id, doc in corpus.items():
                stem = self._safe_stem(doc_id)
                # Resolve rare collisions by appending a counter.
                candidate = stem
                suffix = 0
                while candidate in stem_to_docid:
                    suffix += 1
                    candidate = f"{stem}_{suffix}"
                stem_to_docid[candidate] = doc_id
                content = (doc.get("title") or "") + "\n" + (doc.get("text") or "")
                (tmppath / f"{candidate}.txt").write_text(content, encoding="utf-8")

            # Step 2: Index.
            logger.info("Indexing with skelesearch (provider=%s)...", self.provider)
            idx = subprocess.run(
                [self.binary, "index", tmpdir, "--provider", self.provider],
                capture_output=True,
                timeout=3600,
                cwd=tmpdir,
            )
            if idx.returncode != 0:
                stderr = idx.stderr.decode(errors="replace")
                logger.error(
                    "Index failed (exit %d): %s", idx.returncode, stderr[:500]
                )
                # Return empty results rather than aborting the whole run so
                # other tasks (if any) can still complete.
                return {qid: {} for qid in queries}

            # Step 3: Search.
            logger.info("Searching %d queries...", len(queries))
            env = os.environ.copy()
            results: Dict[str, Dict[str, float]] = {}

            for qid, query_text in queries.items():
                proc = subprocess.run(
                    [
                        self.binary, "search",
                        query_text[:2000],  # guard against pathologically long queries
                        "--top-k", str(effective_top_k),
                        "--json",
                    ],
                    capture_output=True,
                    timeout=120,
                    cwd=tmpdir,
                    env=env,
                )
                if proc.returncode != 0:
                    logger.warning(
                        "Search failed for qid=%s: %s",
                        qid,
                        proc.stderr.decode(errors="replace")[:200],
                    )
                    results[qid] = {}
                    continue

                try:
                    hits = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON output for qid=%s", qid)
                    results[qid] = {}
                    continue

                qresults: Dict[str, float] = {}
                for hit in hits:
                    # file_path from the binary is relative to cwd (tmpdir).
                    stem = Path(hit.get("file_path", "")).stem
                    doc_id = stem_to_docid.get(stem)
                    if doc_id is not None:
                        score = float(hit.get("score", 0.0))
                        # A corpus document could span multiple chunks; keep best.
                        if doc_id not in qresults or qresults[doc_id] < score:
                            qresults[doc_id] = score
                results[qid] = qresults

        return results


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def resolve_task_names(raw: List[str]) -> List[str]:
    """Expand 'all', validate names, and return the final list."""
    if "all" in raw:
        return list(ALL_TASK_NAMES)
    unknown = [t for t in raw if t not in ALL_TASK_NAMES]
    if unknown:
        logger.error(
            "Unknown task(s): %s\nAvailable: %s",
            ", ".join(unknown),
            ", ".join(ALL_TASK_NAMES),
        )
        sys.exit(1)
    return raw


def load_tasks(task_names: List[str]) -> Dict:
    """Download and return {task_name: (corpus, queries, qrels)} from HuggingFace."""
    from coir.data_loader import get_tasks  # type: ignore[import-untyped]

    logger.info("Loading task data from HuggingFace: %s", ", ".join(task_names))
    tasks = get_tasks(task_names)
    if not tasks:
        logger.error(
            "No tasks loaded.  Check task names and network connectivity.\n"
            "Run with --list-tasks to see valid names."
        )
        sys.exit(1)
    logger.info(
        "Loaded %d task(s): %s", len(tasks), ", ".join(sorted(tasks.keys()))
    )
    return tasks


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def run_embed_only(
    model: EmbedOnlyModel,
    tasks: Dict,
    tasks_output_dir: Path,
    batch_size: int,
) -> Dict[str, Dict]:
    """Run embed-only evaluation via the COIR class (wraps DRES internally)."""
    from coir.evaluation import COIR  # type: ignore[import-untyped]

    tasks_output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = COIR(tasks, batch_size=batch_size)
    raw = evaluator.run(model, output_folder=str(tasks_output_dir))

    # COIR.run() skips tasks whose per-task JSON already exists but doesn't
    # reload them into its return value.  Load cached results so the caller
    # always gets the full picture, enabling clean resume behaviour.
    for task_name in tasks:
        if task_name not in raw:
            cached = tasks_output_dir / f"{task_name}.json"
            if cached.exists():
                with open(cached) as f:
                    raw[task_name] = json.load(f).get("metrics", {})

    return raw


def run_hybrid(
    backend: HybridSearch,
    tasks: Dict,
    tasks_output_dir: Path,
    top_k: int,
) -> Dict[str, Dict]:
    """Run hybrid evaluation via EvaluateRetrieval + HybridSearch."""
    from coir.beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore[import-untyped]
    from coir.beir.retrieval.search.base import BaseSearch  # type: ignore[import-untyped]

    # Wrap HybridSearch in an ABC-conforming adapter so EvaluateRetrieval is
    # satisfied without coupling the class definition to coir imports at
    # module-load time.
    class _Adapter(BaseSearch):
        def search(self, corpus, queries, top_k, **kwargs):
            return backend.search(corpus, queries, top_k, **kwargs)

    tasks_output_dir.mkdir(parents=True, exist_ok=True)
    retriever = EvaluateRetrieval(_Adapter(), k_values=[1, 3, 5, 10, 100])
    raw: Dict[str, Dict] = {}

    for task_name, (corpus, queries, qrels) in tasks.items():
        cached = tasks_output_dir / f"{task_name}.json"
        if cached.exists():
            logger.info("Task '%s': loading cached results.", task_name)
            with open(cached) as f:
                raw[task_name] = json.load(f).get("metrics", {})
            continue

        logger.info(
            "Task '%s': hybrid search over %d docs / %d queries...",
            task_name, len(corpus), len(queries),
        )
        t0 = time.time()
        retrieved = retriever.retrieve(corpus, queries)
        ndcg, map_, recall, precision = retriever.evaluate(
            qrels, retrieved, retriever.k_values
        )
        elapsed = time.time() - t0
        logger.info("Task '%s' done in %.1fs.", task_name, elapsed)

        metrics = {"NDCG": ndcg, "MAP": map_, "Recall": recall, "Precision": precision}
        with open(cached, "w") as f:
            json.dump({"metrics": metrics}, f, indent=2)
        raw[task_name] = metrics

    return raw


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def flatten_metrics(nested: Dict) -> Dict[str, float]:
    """Flatten COIR's nested metrics into {metric_name: value}.

    COIR returns:
      {"NDCG": {"NDCG@10": 0.25, ...}, "MAP": {...}, "Recall": {...}, ...}

    We flatten to:
      {"NDCG@10": 0.25, "MAP@10": ..., ...}
    """
    flat: Dict[str, float] = {}
    for _category, values in nested.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[_category] = values
    return flat


def build_output(
    mode: str,
    provider: str,
    raw_results: Dict[str, Dict],
) -> Dict:
    """Produce the final aggregated JSON in the required schema."""
    task_metrics: Dict[str, Dict[str, float]] = {}
    ndcg10_values: List[float] = []

    for task_name, metrics in raw_results.items():
        flat = flatten_metrics(metrics)
        task_metrics[task_name] = flat
        if "NDCG@10" in flat:
            ndcg10_values.append(flat["NDCG@10"])

    avg = (
        round(sum(ndcg10_values) / len(ndcg10_values), 6)
        if ndcg10_values
        else None
    )

    return {
        "benchmark": "CoIR",
        "mode": mode,
        "provider": provider,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks_evaluated": sorted(task_metrics.keys()),
        "results": task_metrics,
        "average_nDCG@10": avg,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EPILOG = """
Modes:
  embed-only   Measures embedding quality via cosine similarity search.
               No skelesearch binary needed; uses the provider's Python API.
               Scores are directly comparable to published MTEB/BEIR numbers.

  hybrid       Measures the full skelesearch pipeline: index + BM25 + vector +
               RRF + optional reranker.  Requires --binary.

Available tasks (--tasks accepts comma-separated names or 'all'):
  Standalone:  cosqa, apps, synthetic-text2sql, codetrans-dl,
               codetrans-contest, codefeedback-st, codefeedback-mt,
               stackoverflow-qa
  Groups:      codesearchnet  (→ 6 CodeSearchNet language sub-tasks)
               codesearchnet-ccr  (→ 6 CCR sub-tasks)
  Special:     all  (runs all standalone + group tasks)

Primary metric: nDCG@10 (higher is better).

Data is downloaded from HuggingFace on first use and cached locally by the
datasets library (typically ~/.cache/huggingface/datasets/).

Examples:
  # Quick smoke test — two small tasks, no API key needed:
  uv run --with coir-eval --with fastembed --with numpy \\
    python3 benchmarks/scripts/coir-eval.py \\
    --mode embed-only --provider fastembed \\
    --tasks cosqa,apps \\
    --output benchmarks/runs/coir-fastembed.json

  # Full embed-only run:
  uv run --with coir-eval --with fastembed --with numpy \\
    python3 benchmarks/scripts/coir-eval.py \\
    --mode embed-only --provider fastembed \\
    --tasks all \\
    --output benchmarks/runs/coir-fastembed-full.json

  # Hybrid mode:
  uv run --with coir-eval --with fastembed --with numpy \\
    python3 benchmarks/scripts/coir-eval.py \\
    --mode hybrid \\
    --binary ./target/release/skelesearch \\
    --provider fastembed \\
    --tasks cosqa \\
    --output benchmarks/runs/coir-hybrid.json
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoIR benchmark adapter for skelesearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument(
        "--mode",
        choices=["embed-only", "hybrid"],
        default="embed-only",
        help="Evaluation mode (default: embed-only)",
    )
    parser.add_argument(
        "--provider",
        default="fastembed",
        choices=["fastembed", "voyage", "openai"],
        help="Embedding provider (default: fastembed)",
    )
    parser.add_argument(
        "--tasks",
        default="cosqa",
        help="Comma-separated task names, or 'all' (default: cosqa)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path for the aggregated results",
    )
    parser.add_argument(
        "--binary",
        default="./target/release/skelesearch",
        help="Path to skelesearch binary (hybrid mode only, default: ./target/release/skelesearch)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size for embed-only mode (default: 64)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Results to retrieve per query in hybrid mode (default: 100)",
    )
    parser.add_argument(
        "--tasks-dir",
        default=None,
        help=(
            "Directory for per-task JSON outputs.  Enables resume: already-evaluated "
            "tasks are loaded from disk and skipped.  Defaults to "
            "<output-stem>-tasks/ next to --output."
        ),
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available task names and exit",
    )
    args = parser.parse_args()

    if args.list_tasks:
        print("Standalone tasks:")
        for t in STANDALONE_TASKS:
            print(f"  {t}")
        print("\nGroup tasks (expand to per-language sub-tasks):")
        for t in GROUP_TASKS:
            print(f"  {t}")
        print("\nSpecial alias:")
        print("  all  — runs all standalone + group tasks")
        sys.exit(0)

    # --output is required unless --list-tasks was given (which exits above).
    if not args.output:
        parser.error("--output is required")

    # Resolve paths.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_output_dir = (
        Path(args.tasks_dir)
        if args.tasks_dir
        else output_path.parent / (output_path.stem + "-tasks")
    )

    # Parse and validate tasks.
    task_names = resolve_task_names(
        [t.strip() for t in args.tasks.split(",") if t.strip()]
    )
    tasks = load_tasks(task_names)

    # Evaluate.
    if args.mode == "embed-only":
        model = EmbedOnlyModel(args.provider)
        raw_results = run_embed_only(model, tasks, tasks_output_dir, args.batch_size)
    else:  # hybrid
        backend = HybridSearch(args.binary, args.provider, top_k=args.top_k)
        raw_results = run_hybrid(backend, tasks, tasks_output_dir, top_k=args.top_k)

    if not raw_results:
        logger.error("No results produced — check tasks and connectivity.")
        sys.exit(1)

    # Write aggregated output.
    output = build_output(args.mode, args.provider, raw_results)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Print summary table.
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  CoIR Results — mode={args.mode}  provider={args.provider}")
    print(sep)
    for task_name in sorted(output["results"]):
        ndcg10 = output["results"][task_name].get("NDCG@10", "n/a")
        print(f"  {task_name:<42} nDCG@10: {ndcg10}")
    if output["average_nDCG@10"] is not None:
        print(f"\n  Average nDCG@10: {output['average_nDCG@10']}")
    print(f"\n  Results written to : {output_path}")
    print(f"  Per-task JSONs in  : {tasks_output_dir}")
    print(sep)


if __name__ == "__main__":
    main()
