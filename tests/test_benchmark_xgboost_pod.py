"""The P2 benchmark, checked against a known answer.

Agent finding, and the reason this file exists: 85 benchmark scripts produce
every number the architecture gate decides on, 6 of them are reachable from
any test, and five of the eight defects the 2026-09-20 review found sat in
that layer. It is the least-tested code in the repository and everything
depends on it.

`measure()` is callable, so the claims the `xgboost_pod` gate check reads
can be verified here instead of being taken on trust from a JSON file.
"""
import pytest

pytest.importorskip("xgboost")

from research.benchmark_xgboost_pod import POD_RAM_BYTES, measure


@pytest.fixture(scope="module")
def measured():
    return measure()


def test_the_pod_is_reached_through_the_factory_by_kind(measured):
    assert measured["registered_kinds"] == ["xgboost"]
    assert measured["created_via_factory"] is True


def test_the_budget_refuses_a_pod_it_has_no_room_for(measured):
    """A budget that cannot refuse is not a budget — the defect this audit
    found in the dream cycle's self-issued RAM lease."""
    assert measured["budget_refused_third_pod"] is True
    assert "budget exhausted" in measured["budget_refusal"]
    assert measured["governor_used_bytes_at_peak"] == 2 * POD_RAM_BYTES


def test_two_independently_activated_pods_answer_identically(measured):
    assert measured["deterministic_across_pods"] is True
    assert measured["deterministic_across_calls"] is True


def test_the_determinism_claim_is_not_vacuous(measured):
    """A model that answered the same class for every probe would satisfy
    every determinism assertion above while proving nothing."""
    assert measured["distinct_predictions"] >= 2


def test_release_gives_every_byte_back_and_frees_the_pod_id(measured):
    assert measured["governor_used_bytes_after_release"] == 0
    assert measured["released_to_zero"] is True
    assert measured["reactivation_after_release"] is True


def test_the_measurement_is_reproducible():
    """Same seed, same numbers. Evidence that changes between two runs of
    the same code cannot be compared against a threshold at all."""
    first, second = measure(), measure()
    assert first["sample_answer"] == second["sample_answer"]
    assert first["training"]["model_bytes"] == second["training"]["model_bytes"]
