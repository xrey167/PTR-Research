"""Small reproducible benchmark for the self-hosted retrieval/manifest path."""
from __future__ import annotations

import json
import argparse
import os
import platform
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_pods.execution_manifest import ExecutionManifest
from neural_pods.local_search import LocalSearchBackend
from neural_pods.search_agent import LocalSearchAgent
from neural_pods.registry import Registry


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


def main(output="runs/local-stack-benchmark-001/report.json", corpus_size=2000):
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="neural-pods-bench-") as td:
        backend = LocalSearchBackend(Path(td) / "search.sqlite")
        backend.create_namespace("bench")
        for i in range(corpus_size):
            supplier = "muller" if i == 1999 else f"supplier-{i % 20}"
            text = f"{supplier} X12 lead time {24 if i == 1999 else i % 60} days record {i}"
            backend.upsert("bench", f"doc-{i}", text=text, vector=[1.0, 0.0] if i == 1999 else [0.0, 1.0],
                           metadata={"status": "active", "acl": ["buyer"], "supplier": supplier})
        query = dict(namespace="bench", text="Muller X12 lead time", vector=[1.0, 0.0],
                     filters={"supplier": "muller"}, principal="buyer", top_k=5)
        warm = []
        for _ in range(100):
            t = time.perf_counter(); hits = backend.search(**query); warm.append(time.perf_counter() - t)
            assert hits and hits[0].key == "doc-1999"
        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: backend.search(**query), range(800)))
        parallel_elapsed = time.perf_counter() - parallel_started
        assert all(x and x[0].key == "doc-1999" for x in results)
        agent = LocalSearchAgent(backend, namespace="bench", parallelism=8, max_turns=2)
        episodes = []
        for _ in range(50):
            episode = agent.run("Muller X12 lead time", ["doc-1999"])
            episodes.append(episode)
            assert episode.recall == 1.0
        adaptive_episodes = []
        for _ in range(50):
            episode = agent.run("Muller X12 lead time", ["doc-1999"], policy=LocalSearchAgent.adaptive_policy)
            adaptive_episodes.append(episode)
            assert episode.recall == 1.0
        backend.close()
        registry = Registry(Path(td) / "registry.sqlite")
        origin = registry.origin("bench", "record", "1", {"value": 24}, acl=["buyer"])
        generation = registry.publish("bench:fact", {"value": 24}, [origin], principal="buyer", acl=["buyer"])
        artifact = registry.artifact("vector", {"embedding": [1.0]}, [generation], principal="buyer")
        manifest_times = []
        manifest = ExecutionManifest.build(registry, generation, [artifact], principal="buyer")
        for _ in range(1000):
            t = time.perf_counter(); manifest.validate(registry); manifest_times.append(time.perf_counter() - t)
        registry.close()
    report = {
        "schema": "local-stack-benchmark:v1", "status": "completed",
        "host": {"platform": platform.platform(), "python": platform.python_version(), "cpu": platform.processor()},
        "corpus": corpus_size, "sequential_searches": 100, "parallel_searches": 800, "parallel_workers": 8,
        "search_latency_s": {"p50": statistics.median(warm), "p95": percentile(warm, .95), "max": max(warm)},
        "parallel_elapsed_s": parallel_elapsed, "parallel_qps": 800 / parallel_elapsed,
        "episodes": 50, "episode_recall": statistics.mean(e.recall for e in episodes),
        "episode_mrr": statistics.mean(e.mrr for e in episodes),
        "episode_latency_s": {"p50": statistics.median(e.elapsed_s for e in episodes), "p95": percentile([e.elapsed_s for e in episodes], .95)},
        "adaptive_teacher": {"episodes": 50, "recall": statistics.mean(e.recall for e in adaptive_episodes),
                              "mrr": statistics.mean(e.mrr for e in adaptive_episodes),
                              "search_calls_mean": statistics.mean(e.search_calls for e in adaptive_episodes),
                              "latency_p50_s": statistics.median(e.elapsed_s for e in adaptive_episodes)},
        "manifest_validations": 1000,
        "manifest_validation_s": {"p50": statistics.median(manifest_times), "p95": percentile(manifest_times, .95), "max": max(manifest_times)},
        "elapsed_total_s": time.perf_counter() - started,
        "limitations": [f"exact vector scan; synthetic {corpus_size}-row corpus; local SQLite only; no ROCm/vLLM server"],
    }
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=int, default=2000)
    parser.add_argument("--output", default="runs/local-stack-benchmark-001/report.json")
    args = parser.parse_args()
    main(args.output, args.corpus)
