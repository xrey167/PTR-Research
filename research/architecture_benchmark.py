"""Repeatable local architecture gates for Pod search and cache behavior."""
from __future__ import annotations
import argparse, json, statistics, tempfile, time
from pathlib import Path
from neural_pods.local_search import LocalSearchBackend
from neural_pods.pod_cache import PodCache


def percentile(values, p):
    values = sorted(values)
    if not values: return 0.0
    return values[min(len(values) - 1, int(round((p / 100) * (len(values) - 1))))]


def run(rows: int = 1000, queries: int = 100):
    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalSearchBackend(Path(tmp) / "bench.sqlite")
        backend.create_namespace("bench")
        for i in range(rows):
            backend.upsert("bench", str(i), text=f"supplier muller x12 lead time item {i}",
                           vector=[1.0, 0.0], metadata={"status": "active", "type": "context",
                           "tags": ["supplier"], "acl": ["buyer"]})
        cache = PodCache(ttl_seconds=120)
        latencies = []
        for _ in range(queries):
            started = time.perf_counter()
            cache.search(backend, "bench", text="muller x12", vector=[1.0, 0.0],
                         principal="buyer", top_k=20)
            latencies.append((time.perf_counter() - started) * 1000)
        # ACL leakage gate
        hidden = cache.search(backend, "bench", text="muller", principal="outsider", top_k=20)
        acl_leakage = len(hidden)
        backend.branch("bench", target="experiment")
        backend.upsert("bench", "new", text="new branch fact", vector=[1, 0],
                       branch="experiment", metadata={"status": "active", "acl": ["buyer"]})
        branch_visible = any(h.key == "new" for h in backend.search("bench", branch="experiment", text="new", principal="buyer"))
        stale_generation = False
        backend.upsert("bench", "generation", text="generation g8", metadata={"status": "active", "generation_key": "g8"})
        old = cache.search(backend, "bench", text="generation", filters={"generation_key": "g7"}, principal="buyer")
        stale_generation = not old
        backend.close()
        return {"rows": rows, "queries": queries, "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "p99": percentile(latencies, 99)},
                "cache": dict(cache.stats()), "acl_leakage_hits": acl_leakage,
                "branch_update_visible": branch_visible, "stale_generation_rejected": stale_generation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--rows", type=int, default=1000); parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.rows, args.queries), indent=2))
