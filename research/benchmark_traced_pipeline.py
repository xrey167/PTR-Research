"""Broad traced pipeline test: all 132 frozen test cases flow through the
complete pod pipeline — cache check -> mesh lookup -> answer (frozen Gen-7
evidence) -> native MQTT frame publication — with a trace record per hop
(JSONL: trace_id, case, stage, latency_ms, flags).

Both runs start from an invalidated cache. Run 1 issues the mesh lookups
one at a time (publish, wait, next); run 2 fires all lookups of the cache
misses first and collects the replies afterwards. The difference measured
is pipelining, not parallelism inside a single lookup.

Every stage latency below is measured. An earlier version appended the
constant 0.05 for run 2's lookup stage and wrote latency_ms = -1 into the
trace, so the reported "20.5 ms -> 0.05 ms" was not a measurement; the
per-case time from publish to reply is now recorded for both runs, which
makes them comparable.
"""
import json
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.mesh import MeshEndpoint  # noqa: E402
from neural_pods.mesh_cache import MeshCache  # noqa: E402
from neural_pods.native_comm import parse_frames  # noqa: E402
from research.evidence import write as write_evidence  # noqa: E402
from research.train_reader import file_sha, load_bundle  # noqa: E402

#: The modules these numbers are evidence ABOUT. research/evidence.py
#: hashes them into the report, and the gate refuses the file once any
#: of them changes: a measurement of code that no longer exists is not
#: evidence, however carefully it was recorded.
SUBJECT = [
    "neural_pods/mesh.py",
    "neural_pods/mesh_cache.py",
    "neural_pods/native_comm.py",
]

BROKER = "10.50.0.121"
REMOTE_DELAY_S = 0.02
PROJECT = Path(__file__).resolve().parents[1]


class Tracer:
    def __init__(self, path: Path):
        self.path = path
        self._handle = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def record(self, trace_id: str, case: str, stage: str, ms: float, **flags) -> None:
        with self._lock:
            self._handle.write(json.dumps({
                "trace_id": trace_id, "case": case, "stage": stage,
                "latency_ms": round(ms, 3), **flags}) + "\n")

    def close(self) -> None:
        self._handle.close()


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(len(values) * p))], 3)


def run_pipeline(endpoint, cache, rows, gen7_answers, tracer, run_name,
                 parallel_lookups):
    stage_stats = {"cache": [], "lookup": [], "answer": [], "publish": []}
    cache_hits = 0
    reflex_valid = 0
    traces = []
    # Both runs use the same endpoint, so its reply bookkeeping has to start
    # empty: a case that times out in run 2 would otherwise be timed against
    # run 1's stamp and record a NEGATIVE latency into the traces and the
    # percentiles.
    endpoint._pending_lookup.clear()
    endpoint._answered_at.clear()
    started = time.perf_counter()

    if parallel_lookups:
        # PIPELINED PATH: publish every cache miss's lookup, then collect the
        # replies, instead of one publish-wait-publish hop per case.
        pending = []
        for row in rows:
            trace_id = str(uuid.uuid4())
            case = row["id"]
            t0 = time.perf_counter()
            cached = cache.get(f"reader:{case}")
            cache_ms = (time.perf_counter() - t0) * 1000
            tracer.record(trace_id, case, "cache", cache_ms, hit=cached is not None)
            stage_stats["cache"].append(cache_ms)
            if cached is not None:
                cache_hits += 1
                traces.append((trace_id, case, cached))
            else:
                pending.append((trace_id, case))

        if pending:
            # Batch-async mesh idiom: fire ALL lookups first, then collect.
            # The responder registered in main() already answers each call on
            # its own thread; a second responder on the same topic would make
            # two pods answer every call and muddy the comparison.
            published_at: dict[str, float] = {}
            for _trace_id, case in pending:
                endpoint._pending_lookup[case] = threading.Event()
            for _trace_id, case in pending:
                published_at[case] = time.perf_counter()
                endpoint.publish("call", {"case": case},
                                 target_pod="lookup-responder")
            collect_deadline = time.perf_counter() + 30.0
            while any(not endpoint._pending_lookup[c].is_set()
                      for _t, c in pending) and time.perf_counter() < collect_deadline:
                time.sleep(0.005)
            collect_ended = time.perf_counter()
            for trace_id, case in pending:
                answered = endpoint._pending_lookup[case].is_set()
                # Real per-case latency: publish -> reply, the same definition
                # the sequential path uses, so the two runs are comparable.
                # An unanswered case is timed to the end of the collect loop,
                # never to a stamp left over from an earlier run.
                end = endpoint._answered_at[case] if answered else collect_ended
                lookup_ms = (end - published_at[case]) * 1000
                tracer.record(trace_id, case, "lookup", lookup_ms,
                              parallel=True, answered=answered)
                stage_stats["lookup"].append(lookup_ms)
                cache.put(f"reader:{case}", {"revision": 7})
                # answer + native publish stages (identical to sequential path)
                t2 = time.perf_counter()
                answer = gen7_answers.get(case, "")
                answer_ms = (time.perf_counter() - t2) * 1000
                tracer.record(trace_id, case, "answer", answer_ms,
                              tokens=len(answer.split()))
                stage_stats["answer"].append(answer_ms)
                t3 = time.perf_counter()
                frame = (f'PUB np/reader/answer '
                         f'{json.dumps({"case": case, "answer": answer}, separators=(",", ":"))}')
                frames = parse_frames(frame)
                reflex_valid += 1 if frames and all(f.valid for f in frames) else 0
                publish_ms = (time.perf_counter() - t3) * 1000
                tracer.record(trace_id, case, "publish", publish_ms,
                              frames_valid=frames and all(f.valid for f in frames))
                stage_stats["publish"].append(publish_ms)
        wall = time.perf_counter() - started
    else:
        for row in rows:
            trace_id = str(uuid.uuid4())
            case = row["id"]
            t0 = time.perf_counter()
            cached = cache.get(f"reader:{case}")
            cache_ms = (time.perf_counter() - t0) * 1000
            was_hit = cached is not None
            tracer.record(trace_id, case, "cache", cache_ms, hit=was_hit)
            stage_stats["cache"].append(cache_ms)
            cache_hits += 1 if was_hit else 0
            if not was_hit:
                # mesh lookup via the dedicated lookup responder (delay 20ms)
                event = threading.Event()
                endpoint._pending_lookup[case] = event
                t1 = time.perf_counter()
                endpoint.publish("call", {"case": case},
                                 target_pod="lookup-responder")
                event.wait(timeout=5.0)
                lookup_ms = (time.perf_counter() - t1) * 1000
                tracer.record(trace_id, case, "lookup", lookup_ms,
                              answered=endpoint._pending_lookup[case].is_set())
                stage_stats["lookup"].append(lookup_ms)
                cache.put(f"reader:{case}", {"revision": 7})
                cached = {"revision": 7}

            # answer stage: frozen Gen-7 evidence (the main model pod)
            t2 = time.perf_counter()
            answer = gen7_answers.get(case, "")
            answer_ms = (time.perf_counter() - t2) * 1000
            tracer.record(trace_id, case, "answer", answer_ms,
                          tokens=len(answer.split()))
            stage_stats["answer"].append(answer_ms)

            # native frame publication: dialect frame parsed + ACL-checked
            t3 = time.perf_counter()
            value = 0
            frame = (f'PUB np/reader/answer '
                     f'{json.dumps({"case": case, "answer": answer}, separators=(",", ":"))}')
            frames = parse_frames(frame)
            reflex_valid += 1 if frames and all(f.valid for f in frames) else 0
            publish_ms = (time.perf_counter() - t3) * 1000
            tracer.record(trace_id, case, "publish", publish_ms,
                          frames_valid=frames and all(f.valid for f in frames))
            stage_stats["publish"].append(publish_ms)
        wall = time.perf_counter() - started

    return {"run": run_name, "wall_s": round(wall, 3),
            "cache_hits": cache_hits,
            "cases": len(rows),
            # Renamed from `reflex_frames_valid`, which claimed far more than
            # it measured: the frame is built here with json.dumps and parsed
            # two lines later, so this count equals `cases` by construction.
            # It says the serialiser and the parser agree. It says nothing
            # about the reflex, and nothing about a model producing frames —
            # that is research/benchmark_native_comm.py's exact_rate.
            "serialised_frames_valid": reflex_valid,
            "stage_p50_ms": {k: percentile(v, 0.5) for k, v in stage_stats.items()},
            "stage_p95_ms": {k: percentile(v, 0.95) for k, v in stage_stats.items()}}


def main() -> None:
    import redis
    protocol, _config, _ = load_bundle(PROJECT / "runs/reader-training-inputs-generation7")
    rows = json.loads((PROJECT / "runs/reader-training-inputs-generation7/inputs/test.json")
                      .read_text(encoding="utf-8"))
    gen7 = json.loads((PROJECT / "runs/qwen3b-eval-test-gen7-20260920-report.json")
                      .read_text(encoding="utf-8"))
    gen7_answers = {r["id"]: r["text"] for r in gen7["rows"]}

    endpoint = MeshEndpoint(BROKER, "traced-host", manifest_hash="trace-manifest")
    # Dedicated lookup responder (simulated remote pod, 20 ms processing).
    endpoint._pending_lookup = {}
    endpoint._answered_at = {}

    def _answered(case: str) -> None:
        """Stamp the reply time once, then release the waiter. Both runs read
        this stamp, so their lookup latencies mean the same thing."""
        event = endpoint._pending_lookup.get(case)
        if event is not None and not event.is_set():
            endpoint._answered_at[case] = time.perf_counter()
            event.set()

    def _process(case: str) -> None:
        time.sleep(REMOTE_DELAY_S)
        endpoint.publish_raw("np/lookup-host/reply", {"case": case})
        _answered(case)

    def _on_lookup_call(topic: str, envelope: dict) -> None:
        # Sleep OFF the paho loop thread - otherwise all calls serialize.
        threading.Thread(target=_process,
                         args=(envelope["body"]["case"],), daemon=True).start()

    lookup_responder = MeshEndpoint(BROKER, "lookup-responder",
                                    manifest_hash="trace-manifest")
    lookup_responder.subscribe("np/lookup-responder/call", _on_lookup_call)
    endpoint.subscribe("np/lookup-host/reply",
                       lambda t, e: _answered(e["body"]["case"]))
    cache = MeshCache(redis_client=redis.Redis(host=BROKER, socket_timeout=3),
                      pod_id="traced-host", principal="tenant-trace",
                      namespace="traced")
    tracer = Tracer(PROJECT / "research/runs/traced-pipeline-20260920.jsonl")
    try:
        for row in rows:  # force run 1 to be genuinely cold
            cache.invalidate(f"reader:{row['id']}")
        run1 = run_pipeline(endpoint, cache, rows, gen7_answers, tracer, "cold",
                            parallel_lookups=False)
        # warm the parallel path's cache view (cold run already populated it;
        # invalidate to make run 2 measure the optimized cold path fairly)
        for row in rows:
            cache.invalidate(f"reader:{row['id']}")
        run2 = run_pipeline(endpoint, cache, rows, gen7_answers, tracer,
                            "parallel-lookups", parallel_lookups=True)
        result = {"status": "completed", "cases": len(rows),
                  "runs": [run1, run2],
                  "improvement_factor": round(
                      run1["wall_s"] / max(run2["wall_s"], 1e-9), 2)}
        write_evidence(result, PROJECT / "research/runs/traced-pipeline-20260920.json",
                       __file__, subject=SUBJECT)
        print(json.dumps(result, indent=2))
    finally:
        tracer.close()
        endpoint.close()


if __name__ == "__main__":
    main()
