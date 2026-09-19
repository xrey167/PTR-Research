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
        ranked = sim.dream(request.get("policies", [{"concept_oversample": 3, "lookup_anchor": 2}]))
        return {"winner": ranked[0], "pool_size": len(pool.generations)}

    channel = ReflexChannel(plane, dispatch, default_pod="pod:dream")
    bind_dream_pod(plane, dispatch)

    result = channel.invoke("dream", {"policies": [{"concept_oversample": 3, "lookup_anchor": 2}]})
    assert result["reflex"] is True
    assert result["pod"] == "pod:dream"
    assert result["result"]["winner"]["decisions"] == {"concept_oversample": 3, "lookup_anchor": 2}
    assert channel.stats()["reflex_hits"] == 1
