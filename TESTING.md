# Testing skelesearch

## Prerequisites

### Unit tests — no API key required

```bash
cargo build --features storage-sqlite
```

### Fastembed eval — no API key required

FastEmbed downloads models locally on first run. No credentials needed.

```bash
cargo build --release --features storage-sqlite
```

### Voyage eval — requires API key

```bash
export VOYAGE_API_KEY=<your-key>
cargo build --release --features storage-sqlite
```

## Re-index eval repos

```bash
BINARY=./target/release/skelesearch
for repo in mini-redis hyperfine cobra httpx hono zod; do
  echo "--- $repo ---"
  rm -rf "benchmarks/repos/$repo/.skelesearch"
  RUST_LOG=skelesearch=warn $BINARY index "benchmarks/repos/$repo" --provider voyage
done
```

## Run eval (single repo)

```bash
cd benchmarks/repos/mini-redis
../../../target/release/skelesearch eval ../../../benchmarks/cases/rust/mini-redis.json --provider voyage --json 2>/dev/null \
  | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); a=d['aggregate']; print(f'R@5={a[\"mean_recall_at_5\"]:.1%} MRR={a[\"mean_mrr\"]:.3f}')"
cd ../../..
```

## Run eval (all repos)

```bash
ROOT=$(pwd)
BIN=$ROOT/target/release/skelesearch

for repo in mini-redis hyperfine cobra; do
  echo "=== $repo ==="
  cd "$ROOT/benchmarks/repos/$repo"
  $BIN eval "$ROOT/benchmarks/cases/rust/$repo.json" --provider voyage --json 2>/dev/null \
    | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); a=d['aggregate']; print(f'R@5={a[\"mean_recall_at_5\"]:.1%} MRR={a[\"mean_mrr\"]:.3f} Cases={a[\"total_cases\"]}')"
done

cd "$ROOT/benchmarks/repos/httpx"
echo "=== httpx ==="
$BIN eval "$ROOT/benchmarks/cases/python/httpx.json" --provider voyage --json 2>/dev/null \
  | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); a=d['aggregate']; print(f'R@5={a[\"mean_recall_at_5\"]:.1%} MRR={a[\"mean_mrr\"]:.3f} Cases={a[\"total_cases\"]}')"

for repo in hono zod; do
  echo "=== $repo ==="
  cd "$ROOT/benchmarks/repos/$repo"
  $BIN eval "$ROOT/benchmarks/cases/typescript/$repo.json" --provider voyage --json 2>/dev/null \
    | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); a=d['aggregate']; print(f'R@5={a[\"mean_recall_at_5\"]:.1%} MRR={a[\"mean_mrr\"]:.3f} Cases={a[\"total_cases\"]}')"
done

cd "$ROOT"
```

## Previous baseline

```
R@5=83.3% MRR=0.851 (avg across 6 repos, 240 cases)
```

## Unit tests

```bash
cargo test -p skelesearch-core --lib     # 92 tests
cargo test -p skelesearch-mcp            # 7 tests
```
