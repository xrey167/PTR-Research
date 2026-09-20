import json
import time
from pathlib import Path

import pytest

from neural_pods.dream import (DREAM_CYCLE_EVENT, CycleBudget, HistoryPool,
                               ReplaySimulator, pool_fingerprint, record_cycle)
from neural_pods.registry import Registry


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


# --- the pod contract from DREAM-POD-DESIGN, which the code did not meet ---


def _pool_with(*names):
    pool = HistoryPool()
    cases = _cases()
    for index, name in enumerate(names):
        pool.add_generation(name, {"concept_oversample": index}, "test",
                            _report([("test:topic:0", 1)]), cases, order=index)
    return pool


def test_pool_refuses_a_generation_out_of_order():
    """Safety rule 3: the pool is append-only and a policy is never fitted on
    future outcomes. The coefficients come from CONSECUTIVE pairs, so
    insertion order is the time axis."""
    pool = _pool_with("gen1", "gen2")
    with pytest.raises(ValueError, match="append-only"):
        pool.add_generation("gen0", {"concept_oversample": 9}, "test",
                            _report([("test:topic:0", 1)]), _cases(), order=0)


def test_pool_refuses_a_duplicate_generation():
    pool = _pool_with("gen1")
    with pytest.raises(ValueError, match="already in the pool"):
        pool.add_generation("gen1", {}, "test", _report([]), _cases())


def test_cycle_budget_counts_the_event_log_and_blocks_when_spent():
    """Safety rule 4: self-directed cycles consume a quota visible in the
    event log. The log is the counter — there is no second place."""
    registry = Registry(":memory:")
    budget = CycleBudget(registry, max_per_window=2)
    assert budget.stats() == {"used": 0, "remaining": 2, "max_per_window": 2,
                              "window_hours": 24.0}
    for _ in range(2):
        with registry.transaction():
            registry.record_event(DREAM_CYCLE_EVENT, {"probe": True})
    assert budget.used() == 2 and budget.remaining() == 0
    with pytest.raises(PermissionError, match="budget exhausted"):
        budget.check()


def test_cycle_budget_ignores_events_outside_the_window():
    registry = Registry(":memory:")
    with registry.transaction():
        registry.record_event(DREAM_CYCLE_EVENT, {"probe": True})
    future = time.time() + 48 * 3600
    budget = CycleBudget(registry, max_per_window=1, clock=lambda: future)
    assert budget.used() == 0
    budget.check()                      # the old cycle no longer counts


def test_a_cycle_is_written_into_the_provenance_log():
    """Pod contract: every cycle is an artifact carrying the pool it read,
    the candidates it weighed and the policy it picked."""
    registry = Registry(":memory:")
    pool = _pool_with("gen1", "gen2")
    sim = ReplaySimulator(pool)
    ranked = sim.dream([{"concept_oversample": 1}], allow_extrapolation=True)
    payload = record_cycle(registry, pool=pool, ranked=ranked, winner=ranked[0],
                           backtest=sim.backtest(),
                           lease={"tier": "ram", "amount_bytes": 1024})

    events = registry.events(action=DREAM_CYCLE_EVENT)
    assert len(events) == 1
    recorded = events[0]["payload"]
    assert recorded["generations"] == ["gen1", "gen2"]
    assert recorded["candidates"] == [{"concept_oversample": 1}]
    assert recorded["winner"] == ranked[0]["decisions"]
    assert recorded["lease"] == {"tier": "ram", "amount_bytes": 1024}
    assert recorded["backtest_mode"] == "leave_one_generation_out"
    assert recorded["pool_fingerprint"] == payload["pool_fingerprint"]
    assert events[0]["ts"] is not None       # the log has a time axis now


def test_the_pool_fingerprint_covers_outcomes_not_just_names():
    """Two cycles over the same generation names but different measured
    outcomes must not share an identity."""
    first = _pool_with("gen1", "gen2")
    second = HistoryPool()
    second.add_generation("gen1", {"concept_oversample": 0}, "test",
                          _report([("test:topic:0", 1)]), _cases(), order=0)
    second.add_generation("gen2", {"concept_oversample": 1}, "test",
                          _report([("test:topic:0", 0)]), _cases(), order=1)
    assert pool_fingerprint(first) != pool_fingerprint(second)


def test_a_refused_generation_does_not_advance_the_pool_order():
    """`_max_order` used to be written before the duplicate-name check, so a
    refused call left the time axis moved on. The next legitimate generation
    was then rejected for taking a position nothing had ever occupied."""
    pool = HistoryPool()
    pool.add_generation("gen1", {"concept_oversample": 0}, "test",
                        _report([("test:topic:0", 1)]), _cases(), order=0)

    with pytest.raises(ValueError, match="already in the pool"):
        pool.add_generation("gen1", {"concept_oversample": 1}, "test",
                            _report([("test:topic:0", 1)]), _cases(), order=1)

    # Order 1 is still free, because the refused call took nothing.
    pool.add_generation("gen2", {"concept_oversample": 1}, "test",
                        _report([("test:topic:0", 0)]), _cases(), order=1)
    assert [generation["name"] for generation in pool.generations] == ["gen1", "gen2"]


def test_the_cycle_budget_reads_the_registrys_clock_not_the_wall_clock():
    """Events are stamped with the registry's clock. A budget that computes
    its window from `time.time()` therefore sees every event of a registry
    with an injected clock as outside the window, and the quota never binds —
    which is the opposite of what a safety quota is for."""
    import datetime as dt

    long_ago = dt.datetime(2025, 8, 15, 12, 0, tzinfo=dt.timezone.utc)
    registry = Registry(":memory:", clock=lambda: long_ago)
    pool = _pool_with("gen1", "gen2")
    sim = ReplaySimulator(pool)
    ranked = sim.dream([{"concept_oversample": 1}], allow_extrapolation=True)
    record_cycle(registry, pool=pool, ranked=ranked, winner=ranked[0],
                 backtest=sim.backtest())

    budget = CycleBudget(registry, max_per_window=1)
    assert budget.used() == 1
    assert budget.remaining() == 0
    with pytest.raises(PermissionError, match="budget exhausted"):
        budget.check()


def test_an_aborted_cycle_is_recorded_and_counts_against_the_quota():
    """A cycle that reads the pool, runs the backtest and dreams every
    candidate has spent the resources the quota exists to bound. Recording
    only the cycles that reached a winner let a failing run repeat without
    limit while the quota kept reading zero."""
    registry = Registry(":memory:")
    pool = _pool_with("gen1", "gen2")
    sim = ReplaySimulator(pool)

    payload = record_cycle(registry, pool=pool, ranked=None, winner=None,
                           backtest=sim.backtest(), status="aborted",
                           reason="no candidate policy is identified")
    assert payload["status"] == "aborted"
    assert payload["winner"] is None
    assert payload["generations"] == ["gen1", "gen2"]
    assert payload["reason"].startswith("no candidate policy")

    budget = CycleBudget(registry, max_per_window=1)
    assert budget.used() == 1
    with pytest.raises(PermissionError):
        budget.check()


def test_a_cycle_that_never_started_records_nothing_but_still_parses():
    """The abort path may fire before the pool exists. The event must still
    be writable, because that is the case the quota most needs to see."""
    registry = Registry(":memory:")
    payload = record_cycle(registry, status="aborted",
                           reason="history pool needs at least two generations")
    assert payload["pool_fingerprint"] is None
    assert payload["generations"] == []
    assert payload["candidates"] == []
    assert CycleBudget(registry, max_per_window=8).used() == 1
