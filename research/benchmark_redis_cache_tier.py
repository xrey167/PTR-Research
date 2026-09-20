"""Redis L2 cache-aside tier benchmark against LocalSearchBackend.

Warm phase populates the LRU+Redis tiers; the hot phase re-reads every
query (L2-only hits prove the Redis path). Compares p50/p99 with the LRU-only
variant on the same backend. Redis lives on a separate LXD node (real network).

This is the file the number "L1 Redis 0,3 ms" was attributed to. It never
produced that number: the measurement is p50 0.384 ms / p99 0.757 ms, and
0.30 ms is the mesh presence RTT from a different benchmark over a different
layer. `summarise()` is pure and tested, so the arithmetic behind the number
can be checked without a Redis server — which is what it took for the
misattribution to go unnoticed for as long as it did.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.pod_cache import PodCache
from neural_pods.local_search import LocalSearchBackend
from research.evidence import write as write_evidence

#: The modules these numbers are evidence ABOUT.
SUBJECT = ["neural_pods/pod_cache.py"]


def percentile(values: list[float], p: float) -> float | None:
    """The repository's convention. None for an empty sample: a 0.0 here
    would read as "very fast" instead of "never measured"."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def summarise_tier(stats: dict, hot_latencies: list[float], *,
                   warm_stats: dict | None = None) -> dict:
    """One cache tier's hot-phase result. Pure."""
    return {
        'stats': stats,
        'warm_stats': warm_stats or {},
        'hot_samples': len(hot_latencies),
        'hot_p50_ms': percentile(hot_latencies, 0.5),
        'hot_p99_ms': percentile(hot_latencies, 0.99),
    }


def main():
    import redis

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--redis-host', default='10.50.0.121')
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=Path('research/runs/redis-cache-20260919.json'))
    args = parser.parse_args()

    rng = random.Random(7)
    warm = [f"bench query warm {i}" for i in range(args.count)]
    hot = list(warm)
    rng.shuffle(hot)

    backend = LocalSearchBackend('/tmp/redis-bench.sqlite3')
    backend.create_namespace("bench")
    for i in range(20):
        backend.upsert("bench", f"doc-{i}", text=f"lorem ipsum bench body number {i} for cache tier testing")
    results = {}

    for label, make_cache in (('lru_only', lambda: PodCache(max_entries=args.count, ttl_seconds=600)),
                              ('lru_redis', lambda: PodCache(max_entries=args.count, ttl_seconds=600,
                                                             redis_client=redis.Redis(host=args.redis_host, socket_connect_timeout=3, socket_timeout=3),
                                                             redis_ttl_seconds=600))):
        cache = make_cache()
        for text in warm:
            cache.search(backend, "bench", text=text, top_k=5)
        l1_stats = cache.stats()
        # Fresh LRU, same Redis: proves L2-only hits survive an LRU restart.
        if label == 'lru_redis':
            cache = make_cache()
        hot_lat = []
        for text in hot:
            start = time.perf_counter()
            cache.search(backend, "bench", text=text, top_k=5)
            hot_lat.append((time.perf_counter() - start) * 1000)
        results[label] = summarise_tier(cache.stats(), hot_lat,
                                        warm_stats=l1_stats)

    write_evidence(results, args.output, __file__, subject=SUBJECT)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
