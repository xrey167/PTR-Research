"""LAN benchmark for the two-replica vLLM router."""
from __future__ import annotations

import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from neural_pods.vllm_router import VllmReplica, VllmReplicaRouter


def run(host: str = "192.168.1.223", count: int = 128, workers: int = 16) -> dict:
    router = VllmReplicaRouter([
        VllmReplica("gpu0", f"http://{host}:18000"),
        VllmReplica("gpu1", f"http://{host}:18001"),
    ], timeout_s=60)
    payload = {"model": "neohorse-1-4b", "messages": [{"role": "user", "content": "Answer in one short sentence: what is a pod?"}], "max_tokens": 8, "temperature": 0}

    def one(_index: int):
        started = time.perf_counter()
        try:
            result = router.chat(payload)
            return time.perf_counter() - started, result.get("usage", {}).get("completion_tokens", 0), True
        except Exception:
            return time.perf_counter() - started, 0, False

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, range(count)))
    elapsed = time.perf_counter() - started
    return {"host": host, "count": count, "workers": workers, "elapsed_s": elapsed,
            "requests_per_s": count / elapsed, "tokens_per_s": sum(r[1] for r in rows) / elapsed,
            "ok": sum(r[2] for r in rows), "errors": sum(not r[2] for r in rows),
            "p50_ms": statistics.median(r[0] for r in rows) * 1000,
            "p95_ms": sorted(r[0] for r in rows)[int(.95 * len(rows)) - 1] * 1000,
            "health": router.health(), "metrics": router.metrics.__dict__}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=lambda value: value.__dict__))
