"""Measure local micro-batching overhead and queueing throughput."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from neural_pods.adaptive_batcher import AdaptiveBatcher


def main(count: int, workers: int, output: Path) -> None:
    batches: list[int] = []

    def run_batch(payloads, _key):
        batches.append(len(payloads))
        return payloads

    batcher = AdaptiveBatcher(run_batch, max_batch_size=64, max_wait_ms=2, max_queue=count * 2)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(lambda n=n: batcher.submit(n, key="model:v1").result(timeout=10), n)
                   for n in range(count)]
        results = [future.result(timeout=10) for future in futures]
    elapsed = time.perf_counter() - started
    stats = batcher.stats()
    batcher.close()
    report = {
        "requests": count, "workers": workers, "elapsed_s": elapsed,
        "requests_per_s": count / elapsed, "correct": results == list(range(count)),
        "batches": len(batches), "mean_batch": sum(batches) / max(len(batches), 1),
        "max_batch": max(batches, default=0), "stats": stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.count, args.workers, args.output)
