"""D3: the dream pod bound into the reflex channel.

The main model can ask the dream pod for the current best curriculum
strategy via `invoke("dream", {...})` — no JSON tool-call, just the
learned alias. The dispatch handler runs the replay simulator against the
history pool and returns the ranked strategies.
"""
import json
from pathlib import Path

from neural_pods.registry import Registry
from neural_pods.symlink import TemporalPortPlane
from neural_pods.reflex import ReflexChannel
from neural_pods.dream import HistoryPool, ReplaySimulator


def bind_dream_pod(plane: TemporalPortPlane, dispatch) -> None:
    registry = plane.registry
    origin = registry.origin("pods", "dream", 1, {"kind": "dream_pod"})
    registry.publish("adapter:dream", {"pod_type": "dream"}, parents=[origin])
    plane.bind("adapter:dream", ["dream"], value_handle="pod:dream")


def test_dream_pod_reflex_binding_returns_strategy():
    plane = TemporalPortPlane(Registry(":memory:"))
    plane.registry  # bound below

    pool = HistoryPool()
    cases = [{"id": f"test:lookup:{i}", "evidence": {"lead_time_days": 5}, "target": "5 days",
              "question": f"q{i}"} for i in range(4)] + \
             [{"id": f"test:topic:{i}", "evidence": None, "target": f"c{i}",
               "question": f"c{i}"} for i in range(4)]
    pool.add_generation("gen1", {"concept_oversample": 0, "lookup_anchor": 0}, "test",
                        {"rows": [{"id": "test:lookup:0", "exact_target_match": 1}]}, cases)
    pool.add_generation("gen2", {"concept_oversample": 3, "lookup_anchor": 0}, "test",
                        {"rows": [{"id": "test:lookup:0", "exact_target_match": 1}]}, cases)

    def dispatch(pod_key, request):
        assert pod_key == "pod:dream"
        sim = ReplaySimulator(pool)
        # lookup_anchor 2 was never observed in this toy history; the reflex
        # path dreams past it on purpose, so it says so explicitly.
        ranked = sim.dream(request.get("policies", [{"concept_oversample": 3, "lookup_anchor": 2}]),
                           allow_extrapolation=True)
        return {"winner": ranked[0], "pool_size": len(pool.generations)}

    channel = ReflexChannel(plane, dispatch, default_pod="pod:dream")
    bind_dream_pod(plane, dispatch)

    result = channel.invoke("dream", {"policies": [{"concept_oversample": 3, "lookup_anchor": 2}]})
    assert result["reflex"] is True
    assert result["pod"] == "pod:dream"
    assert result["result"]["winner"]["decisions"] == {"concept_oversample": 3, "lookup_anchor": 2}
    assert channel.stats()["reflex_hits"] == 1


# ---------------------------------------------------------------------------
# The benchmark behind the `dream_reflex` gate check, exercised directly.
# It needs no server, so the one thing that could not be checked before was
# simply that nobody had called it from a test.


def test_the_dream_reflex_benchmark_produces_the_fields_the_gate_reads():
    from research.benchmark_dream_reflex import measure

    result = measure()
    assert result["status"] == "completed"
    assert result["miss_retracted_to_default"] is True
    assert result["winner_deterministic"] is True
    assert result["resolve_within_target"] is True
    assert result["reflex"]["all_misses_covered"] is True
    assert result["reflex"]["errors"] == 0
    # 200 invocations plus one deliberate miss.
    assert result["invocations"] == 201
    assert result["dispatched"] == {"dream": 200, "default": 1}


def test_the_benchmark_says_which_pool_it_measured():
    """A latency measured over a two-generation toy pool is not a latency
    measured over the real one, and the gate now refuses evidence that does
    not say which it was."""
    from research.benchmark_dream_reflex import measure

    result = measure()
    assert result["pool_source"]
    assert result["pool_source"].startswith(("recorded", "synthetic"))
    assert len(result["pool_generations"]) >= 2


def test_resolve_within_target_is_not_satisfied_by_a_missing_measurement():
    """`(p95 or 999) < 5.0` turned a legitimate 0.0 into a failure, and the
    opposite mistake — treating a missing measurement as fast — is the one
    that matters. None must not pass."""
    from research.benchmark_dream_reflex import measure

    result = measure()
    assert result["reflex"]["resolve_p95_ms"] is not None
    assert result["resolve_within_target"] == (
        result["reflex"]["resolve_p95_ms"] < result["latency_target_ms"])
