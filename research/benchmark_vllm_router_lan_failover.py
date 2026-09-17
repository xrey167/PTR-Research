"""LAN failover benchmark; run after one remote replica is stopped."""
from __future__ import annotations
import concurrent.futures, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from neural_pods.vllm_router import VllmReplica, VllmReplicaRouter

router = VllmReplicaRouter([
    VllmReplica("gpu0", "http://192.168.1.223:18000"),
    VllmReplica("gpu1", "http://192.168.1.223:18001"),
], timeout_s=1.0)
payload = {"model": "neohorse-1-4b", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 4, "temperature": 0}

def one(_):
    try:
        router.chat(payload)
        return 1
    except Exception:
        return 0

started = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    rows = list(pool.map(one, range(32)))
elapsed = time.perf_counter() - started
print(json.dumps({"count": 32, "elapsed_s": elapsed, "ok": sum(rows), "errors": len(rows)-sum(rows), "requests_per_s": 32/elapsed, "health": router.health(), "metrics": router.metrics.__dict__}, default=lambda x: x.__dict__))
