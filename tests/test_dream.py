import json
from pathlib import Path

from neural_pods.dream import HistoryPool, ReplaySimulator


def _report(rows):
    return {"rows": [{"id": r[0], "exact_target_match": r[1]} for r in rows]}


def _cases():
    cases = []
    for i in range(4):
        cases.append({"id": f"test:lookup:{i}", "evidence": {"lead_time_days": 5},
                      "target": "5 days", "question": f"q{i}"})
    for i in range(4):
        cases.append({"id": f"test:topic:{i}", "evidence": None,
                      "target": f"concept sentence {i}", "question": f"c{i}"})
    return cases


def test_pool_and_replay_reproduce_recorded_history():
    pool = HistoryPool()
    cases = _cases()
    # gen1: no oversampling — typed 50%, concept 25%
    pool.add_generation("gen1", {"concept_oversample": 0, "lookup_anchor": 0},
                        "test", _report([("test:lookup:0", 1), ("test:lookup:1", 0),
                                         ("test:lookup:2", 1), ("test:lookup:3", 0),
                                         ("test:topic:0", 1), ("test:topic:1", 0),
                                         ("test:topic:2", 0), ("test:topic:3", 0)]),
                        cases)
    # gen2: oversample 2 — typed unchanged, concept 75%
    pool.add_generation("gen2", {"concept_oversample": 2, "lookup_anchor": 0},
                        "test", _report([("test:lookup:0", 1), ("test:lookup:1", 0),
                                         ("test:lookup:2", 1), ("test:lookup:3", 0),
                                         ("test:topic:0", 1), ("test:topic:1", 1),
                                         ("test:topic:2", 1), ("test:topic:3", 0)]),
                        cases)
    assert len(pool.generations) == 2
    deltas = pool.family_deltas()
    assert deltas["concept_oversample"]["concept"] > 0  # oversampling helped concept
    assert deltas["concept_oversample"]["typed"] == 0   # typed untouched

    sim = ReplaySimulator(pool)
    backtest = sim.backtest()
    assert backtest["max_abs_error"] < 0.01  # replay reproduces recorded outcomes


def test_dream_ranks_policies_and_respects_limits():
    pool = HistoryPool()
    cases = _cases()
    pool.add_generation("gen1", {"concept_oversample": 0, "lookup_anchor": 0}, "test",
                        _report([("test:lookup:0", 1), ("test:lookup:1", 1),
                                 ("test:lookup:2", 1), ("test:lookup:3", 1),
                                 ("test:topic:0", 0), ("test:topic:1", 0),
                                 ("test:topic:2", 0), ("test:topic:3", 0)]), cases)
    # sequential decisions keep the effect identifiable: oversample first
    pool.add_generation("gen2", {"concept_oversample": 3, "lookup_anchor": 0}, "test",
                        _report([("test:lookup:0", 1), ("test:lookup:1", 1),
                                 ("test:lookup:2", 1), ("test:lookup:3", 1),
                                 ("test:topic:0", 1), ("test:topic:1", 1),
                                 ("test:topic:2", 1), ("test:topic:3", 1)]), cases)
    sim = ReplaySimulator(pool)
    candidates = [
        {"concept_oversample": 0, "lookup_anchor": 0},
        {"concept_oversample": 3, "lookup_anchor": 0},
        {"concept_oversample": 6, "lookup_anchor": 2},
    ]
    ranked = sim.dream(candidates)
    # The anchor was never observed in this toy history, so (3,0) and (6,2)
    # tie at the top — the simulator correctly refuses to invent effects.
    assert ranked[0]["score"] == max(r["score"] for r in ranked)
    assert ranked[0]["score"] > ranked[-1]["score"]

    try:
        sim.dream([{"concept_oversample": 10, "lookup_anchor": 0}])
        raise AssertionError("expected limits rejection")
    except ValueError as error:
        assert "learned limits" in str(error)


def test_from_project_raises_without_history():
    try:
        HistoryPool.from_project(Path("/tmp/does-not-exist-xyz"))
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "at least two generations" in str(error)
