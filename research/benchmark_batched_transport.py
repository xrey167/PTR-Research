"""End-to-end in-process PodTransport + adaptive batching benchmark."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from neural_pods.adaptive_batcher import BatchedPodHandler
from neural_pods.pod_protocol import PodFanout, PodRequest, PodTransport


def main(count: int, workers: int, output: Path) -> None:
    def run_batch(items, _key):
        return [{"value": payload["value"] * 2} for payload, _request in items]

    handler = BatchedPodHandler(run_batch, max_batch_size=64, max_wait_ms=2)
    transport = PodTransport()
    transport.register("model", "infer", handler)
    requests = [PodRequest("router", "model", "infer", {"value": i},
                           target_generation="g1", target_artifact="a1", principal="bench")
                for i in range(count)]
    started = time.perf_counter()
    responses = PodFanout(transport, max_workers=workers).dispatch(requests)
    elapsed = time.perf_counter() - started
    stats = handler.stats()
    handler.close()
    correct = all(response.ok and response.payload.get("value") == i * 2
                  for i, response in enumerate(responses))
    report = {"requests": count, "workers": workers, "elapsed_s": elapsed,
              "requests_per_s": count / max(elapsed, 1e-9), "responses": len(responses),
              "correct": correct, "errors": sum(not response.ok for response in responses),
              "batch_stats": stats}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.count, args.workers, args.output)
