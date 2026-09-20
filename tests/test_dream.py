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
    deltas, ambiguous = pool.family_deltas()
    assert deltas["concept_oversample"]["concept"] > 0  # oversampling helped concept
    assert deltas["concept_oversample"]["typed"] == 0   # typed untouched
    assert ambiguous == []                              # one decision per transition

    sim = ReplaySimulator(pool)
    backtest = sim.backtest()
    # In sample the fit reproduces its own points - and says so rather than
    # presenting that as accuracy.
    assert backtest["in_sample"]["max_abs_error"] < 0.01
    assert backtest["in_sample"]["degenerate"] is True
    assert backtest["mode"] == "leave_one_generation_out"


def test_in_sample_error_is_zero_by_construction_and_marked_as_such():
    """Two generations, one decision change, two parameters: the model cannot
    miss the points it was fitted on. This is the number that was published
    as "Backtest-Fehler 0,01"."""
    pool = HistoryPool()
    cases = _cases()
    pool.add_generation("gen1", {"concept_oversample": 0}, "test",
                        _report([("test:topic:0", 1), ("test:topic:1", 0),
                                 ("test:topic:2", 0), ("test:topic:3", 0)]), cases)
    pool.add_generation("gen2", {"concept_oversample": 3}, "test",
                        _report([("test:topic:0", 1), ("test:topic:1", 1),
                                 ("test:topic:2", 1), ("test:topic:3", 0)]), cases)
    backtest = ReplaySimulator(pool).backtest()
    assert backtest["in_sample"]["max_abs_error"] == 0.0
    assert backtest["in_sample"]["degenerate"] is True
    # and nothing at all can be predicted out of sample from two points
    assert backtest["out_of_sample"]["generations"] == 0
    assert backtest["out_of_sample"]["max_abs_error"] is None
    reasons = {row["generation"]: row["reason"]
               for row in backtest["out_of_sample"]["per_generation"]}
    assert "intercept" in reasons["gen1"]
    assert "concept_oversample" in reasons["gen2"]


def test_simultaneous_decision_changes_are_not_attributed():
    """A transition that moves two knobs at once tells you nothing about
    either; the old code charged the whole delta to whichever was iterated
    last."""
    pool = HistoryPool()
    cases = _cases()
    pool.add_generation("gen1", {"concept_oversample": 0, "lookup_anchor": 0}, "test",
                        _report([("test:topic:0", 0), ("test:topic:1", 0),
                                 ("test:topic:2", 0), ("test:topic:3", 0)]), cases)
    pool.add_generation("gen2", {"concept_oversample": 3, "lookup_anchor": 1}, "test",
                        _report([("test:topic:0", 1), ("test:topic:1", 1),
                                 ("test:topic:2", 1), ("test:topic:3", 1)]), cases)
    deltas, ambiguous = pool.family_deltas()
    assert deltas == {}
    assert ambiguous == [{"from": "gen1", "to": "gen2",
                          "decisions": ["concept_oversample", "lookup_anchor"]}]
    prediction = ReplaySimulator(pool).simulate({"concept_oversample": 3,
                                                 "lookup_anchor": 1})
    assert prediction["concept"]["estimable"] is False


def test_base_model_change_is_a_modelled_dimension():
    """The Gen-5 -> Gen-6 base-model swap used to fall outside the delta table
    entirely and its effect landed in the residual."""
    pool = HistoryPool()
    cases = _cases()
    pool.add_generation("gen1", {"concept_oversample": 3}, "test",
                        _report([("test:topic:0", 1), ("test:topic:1", 0),
                                 ("test:topic:2", 0), ("test:topic:3", 0)]), cases)
    pool.add_generation("gen2", {"concept_oversample": 3, "base_model": "neohorse"},
                        "test",
                        _report([("test:topic:0", 1), ("test:topic:1", 1),
                                 ("test:topic:2", 1), ("test:topic:3", 0)]), cases)
    deltas, _ = pool.family_deltas()
    assert "base_model=neohorse" in deltas
    assert deltas["base_model=neohorse"]["concept"] > 0


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
    ranked = sim.dream(candidates, allow_extrapolation=True)
    by_policy = {(r["decisions"]["concept_oversample"],
                  r["decisions"]["lookup_anchor"]): r for r in ranked}
    # The anchor never moved in this toy history, so a policy that turns it on
    # is not identified at all: no score, rather than a score that silently
    # treats the unknown decision as having no effect.
    assert by_policy[(6, 2)]["estimable"] is False
    assert by_policy[(6, 2)]["score"] is None
    assert [d["decision"] for d in by_policy[(6, 2)]["extrapolates"]] == [
        "concept_oversample", "lookup_anchor"]
    # Estimable policies rank above unestimable ones, best first.
    assert ranked[0] is by_policy[(3, 0)]
    assert ranked[-1] is by_policy[(6, 2)]
    assert ranked[0]["score"] > by_policy[(0, 0)]["score"]

    try:
        sim.dream([{"concept_oversample": 10, "lookup_anchor": 0}])
        raise AssertionError("expected extrapolation rejection")
    except ValueError as error:
        assert "beyond observed history" in str(error)


def test_extrapolation_is_refused_by_default():
    """The documented intent was "never dream beyond it"; the hard-coded
    limit table allowed lookup_anchor=2 although only 0 and 1 were observed,
    which is how the Gen-7 winner came to be an extrapolation unnoticed."""
    pool = HistoryPool()
    cases = _cases()
    pool.add_generation("gen1", {"lookup_anchor": 0}, "test",
                        _report([("test:lookup:0", 0)]), cases)
    pool.add_generation("gen2", {"lookup_anchor": 1}, "test",
                        _report([("test:lookup:0", 1)]), cases)
    sim = ReplaySimulator(pool)
    assert sim.limits["lookup_anchor"] == (0.0, 1.0)
    try:
        sim.dream([{"lookup_anchor": 2}])
        raise AssertionError("expected extrapolation rejection")
    except ValueError as error:
        assert "lookup_anchor" in str(error)
    flagged = sim.dream([{"lookup_anchor": 2}], allow_extrapolation=True)[0]
    assert flagged["extrapolates"] == [{"decision": "lookup_anchor", "level": 2.0,
                                        "observed": [0.0, 1.0]}]


def test_from_project_raises_without_history():
    try:
        HistoryPool.from_project(Path("/tmp/does-not-exist-xyz"))
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "at least two generations" in str(error)
