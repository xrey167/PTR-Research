"""D3 of the Dream-Pod design: the main model reaches the pod by reflex.

The design marks D3 done and promises a `dream_reflex` gate check. The check
never existed, so the only thing holding the binding up was a unit test. This
benchmark exercises the real path — alias -> TemporalPortPlane -> ReflexChannel
-> Dream-Pod -> policy ranking — and records what it measured.

What it measures: the BINDING and its latency, against the Pod-Arm design's
target of under 5 ms from address signal to dispatch. It says nothing about
whether the dreamed policy is any good; that is what the leave-one-generation-out
backtest is for, and on the recorded history it answers "not measurable".

The history pool is the real one when this runs where the generation reports
live, and a deterministic synthetic pool otherwise. Which one was used is in
the report AND checked by the gate, because a latency measured over a
two-generation toy pool is not a latency measured over the real one. In a
clone that does not carry the generation reports the check is therefore red,
which is the correct verdict: the measurement was not made.
"""
from __future__ import annotations
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.dream import HistoryPool, ReplaySimulator  # noqa: E402
from neural_pods.reflex import ReflexChannel  # noqa: E402
from neural_pods.registry import Registry  # noqa: E402
from neural_pods.symlink import TemporalPortPlane  # noqa: E402
from research.evidence import write as write_evidence  # noqa: E402

#: The modules these numbers are evidence ABOUT. research/evidence.py
#: hashes them into the report, and the gate refuses the file once any
#: of them changes: a measurement of code that no longer exists is not
#: evidence, however carefully it was recorded.
SUBJECT = [
    "neural_pods/dream.py",
    "neural_pods/reflex.py",
    "neural_pods/symlink.py",
    "neural_pods/registry.py",
]

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "research" / "runs" / "dream-reflex-20260920.json"
INVOCATIONS = 200
CANDIDATES = [
    {"concept_oversample": 0, "lookup_anchor": 0},
    {"concept_oversample": 3, "lookup_anchor": 0},
    {"concept_oversample": 3, "lookup_anchor": 1},
]


def _synthetic_pool() -> HistoryPool:
    """Two generations, one decision change: the smallest pool that gives the
    simulator anything to say."""
    cases = ([{"id": f"typed:{i}", "evidence": {"lead_time_days": 5}} for i in range(8)]
             + [{"id": f"concept:{i}", "evidence": None} for i in range(8)])
    pool = HistoryPool()
    for order, (name, oversample, concept_hits) in enumerate(
            [("gen-a", 0, 2), ("gen-b", 3, 6)]):
        rows = ([{"id": f"typed:{i}", "exact_target_match": i < 6} for i in range(8)]
                + [{"id": f"concept:{i}", "exact_target_match": i < concept_hits}
                   for i in range(8)])
        pool.add_generation(name, {"concept_oversample": oversample}, "test",
                            {"rows": rows}, cases, order=order)
    return pool


def _load_pool() -> tuple[HistoryPool, str]:
    try:
        return HistoryPool.from_project(PROJECT, split="test"), "recorded generations"
    except (ValueError, KeyError):
        return _synthetic_pool(), "synthetic (generation reports not in this checkout)"


def measure() -> dict:
    """The measurement, as a function that returns its result.

    Needs no server: the pool comes from the checked-in generation reports
    when they are present and from a deterministic synthetic pool otherwise,
    and the dispatch is in-process. Split out so the verdict fields the gate
    reads can be checked against a known run.
    """
    pool, pool_source = _load_pool()
    simulator = ReplaySimulator(pool)

    registry = Registry(":memory:")
    plane = TemporalPortPlane(registry)
    origin = registry.origin("dream_reflex", "adapter:dream", 1, {"kind": "pod"})
    registry.publish("adapter:dream", {"pod_type": "lora"}, parents=[origin])
    plane.bind("adapter:dream", ["dream"], value_handle="pod:dream")

    dispatched = {"dream": 0, "default": 0}

    def dispatch(pod_key: str, request: dict) -> dict:
        if pod_key == "pod:dream":
            dispatched["dream"] += 1
            ranked = simulator.dream(request["policies"], allow_extrapolation=True)
            return {"winner": ranked[0], "candidates": len(ranked)}
        dispatched["default"] += 1
        return {"winner": None, "deliberate": True}

    channel = ReflexChannel(plane, dispatch, default_pod="pod:deliberate")

    started = time.perf_counter()
    winners = []
    for _ in range(INVOCATIONS):
        result = channel.invoke("dream", {"policies": CANDIDATES})
        winners.append(json.dumps(result["result"]["winner"]["decisions"],
                                  sort_keys=True))
    # One deliberate miss: the arm must retract to the default pod, not fail.
    miss = channel.invoke("no-such-alias", {"policies": CANDIDATES})
    elapsed = time.perf_counter() - started

    stats = channel.stats()
    result = {
        "status": "completed",
        "pool_source": pool_source,
        "pool_generations": [g["name"] for g in pool.generations],
        "invocations": INVOCATIONS + 1,
        "elapsed_s": round(elapsed, 4),
        "reflex": stats,
        "dispatched": dispatched,
        "miss_retracted_to_default": miss["reflex"] is False
                                     and miss["pod"] == "pod:deliberate",
        # A reflex that answers differently each time is not an address, it
        # is a coin flip.
        "winner_deterministic": len(set(winners)) == 1,
        "latency_target_ms": 5.0,
        # `or 999` would have turned a legitimate 0.0 ms into a failure. A
        # MISSING measurement is what must fail here, and only that.
        "resolve_within_target": (stats["resolve_p95_ms"] is not None
                                  and stats["resolve_p95_ms"] < 5.0),
        "scope": ("binding mechanism and its latency only; policy quality is "
                  "the backtest's job, not this benchmark's"),
    }
    return result


def main() -> None:
    result = measure()
    write_evidence(result, OUT, __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
