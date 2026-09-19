"""Redis L2 cache-aside tier benchmark against LocalSearchBackend.

Warm phase populates the LRU+Redis tiers; the hot phase re-reads every
query (L2-only hits prove the Redis path). Compares p50/p99 with the LRU-only
variant on the same backend. Redis lives on a separate LXD node (real network).
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
import redis


def main():
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
                                                             redis_client=redis.Redis(host=args.redis_host),
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
        hot_lat.sort()
        results[label] = {'stats': cache.stats(),
                          'hot_p50_ms': hot_lat[len(hot_lat) // 2],
                          'hot_p99_ms': hot_lat[max(0, int(len(hot_lat) * 0.99) - 1)]}

    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
