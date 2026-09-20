"""The verdicts of the five server-bound mesh benchmarks.

None of these measurements can run here: they need the MQTT broker in
np-node1, a Redis server, and LXD nodes. That is exactly why their
arithmetic went unchecked — and why `mesh_cache` recorded
`principal_isolated: true` from a read of a principal nothing had ever been
written under, and why "0,3 ms" was carried into four documents as a
cross-node latency it never was.

`collect()` still needs the server. `summarise()` does not, and every case
below is one a real run could produce and that the verdict has to get right.
"""
import pytest

from research import benchmark_mesh, benchmark_mesh_cache, benchmark_mesh_e2e
from research import benchmark_native_tcp, benchmark_redis_cache_tier


# --------------------------------------------------------------- mesh presence

def _mesh(rtts, *, discovered=True, rounds_sent=100):
    return dict(discovery={"discovered": discovered, "peer_state": "ready"},
                rtts=rtts, rounds_sent=rounds_sent,
                endpoint_stats={"published": rounds_sent},
                responder_stats={"received": rounds_sent})


def test_mesh_presence_reports_percentiles_and_losses():
    report = benchmark_mesh.summarise(**_mesh([0.30] * 98))
    assert report["rounds_ok"] == 98
    assert report["rounds_lost"] == 2
    assert report["rtt_p50_ms"] == 0.30


def test_mesh_presence_says_the_rtt_is_not_cross_node():
    """Both endpoints run on the host; only the broker is remote. This
    number has been quoted as a cross-node mesh latency in the master
    document and three design documents. The evidence now says otherwise."""
    report = benchmark_mesh.summarise(**_mesh([0.30] * 100))
    assert report["rtt_is_cross_node"] is False
    assert "host endpoint" in report["rtt_scope"]


def test_a_run_where_nothing_came_back_reports_no_latency_not_zero():
    """None, not 0.0: the gate must be able to tell "fast" from "never
    measured", and its threshold is a `<` comparison."""
    report = benchmark_mesh.summarise(**_mesh([]))
    assert report["rtt_p50_ms"] is None
    assert report["rtt_p99_ms"] is None
    assert report["rounds_ok"] == 0


def test_a_single_sample_does_not_index_out_of_range():
    report = benchmark_mesh.summarise(**_mesh([1.5], rounds_sent=1))
    assert report["rtt_p50_ms"] == 1.5
    assert report["rtt_p99_ms"] == 1.5


# ------------------------------------------------------------------ mesh cache

def _peer(**overrides):
    peer = {"shared": {"value": 42, "unit": "days"},
            "cross_principal_refused": True,
            "cross_principal_reachable_via_redis": True,
            "after_invalidate": None}
    peer.update(overrides)
    return peer


def test_mesh_cache_records_both_halves_of_the_isolation_question():
    report = benchmark_mesh_cache.summarise(_peer())
    assert report["cross_node_read"] is True
    assert report["cross_principal_refused_by_client"] is True
    # Expected True. That IS the finding: the entries are not isolated.
    assert report["cross_principal_reachable_via_redis"] is True
    assert report["isolation"] == "client-side key derivation"
    assert "principal_isolated" not in report


def test_a_cache_that_served_a_foreign_principal_is_reported():
    report = benchmark_mesh_cache.summarise(_peer(cross_principal_refused=False))
    assert report["cross_principal_refused_by_client"] is False


def test_a_read_that_returned_the_wrong_value_is_not_a_cross_node_read():
    report = benchmark_mesh_cache.summarise(_peer(shared={"value": 41}))
    assert report["cross_node_read"] is False


def test_an_entry_that_survived_invalidation_is_reported():
    report = benchmark_mesh_cache.summarise(_peer(after_invalidate={"value": 42}))
    assert report["invalidation_works"] is False


# -------------------------------------------------------------------- mesh e2e

def _acks(n, *, validated=True):
    return {i: {"seq": i, "validated": validated} for i in range(n)}


def test_mesh_e2e_requires_every_round_to_be_acknowledged():
    report = benchmark_mesh_e2e.summarise(
        acks=_acks(20), rounds=20, elapsed_s=3.21,
        executor_stats={"refused": 0}, peer_report={})
    assert report["acks_received"] == 20
    assert report["acks_missing"] == 0
    assert report["pod_b_validated_all"] is True


def test_a_run_where_no_ack_arrived_does_not_report_perfect_validation():
    """`all()` over an empty dict is True. Without the count, a run in which
    nothing arrived reported that pod B validated everything."""
    report = benchmark_mesh_e2e.summarise(
        acks={}, rounds=20, elapsed_s=3.21,
        executor_stats={"refused": 0}, peer_report={})
    assert report["pod_b_validated_all"] is False
    assert report["acks_missing"] == 20


def test_a_partial_run_is_not_a_success():
    report = benchmark_mesh_e2e.summarise(
        acks=_acks(19), rounds=20, elapsed_s=3.21,
        executor_stats={"refused": 0}, peer_report={})
    assert report["pod_b_validated_all"] is False


def test_an_ack_that_failed_validation_counts():
    report = benchmark_mesh_e2e.summarise(
        acks=_acks(20) | {5: {"seq": 5, "validated": False}}, rounds=20,
        elapsed_s=3.21, executor_stats={"refused": 0}, peer_report={})
    assert report["pod_b_validated_all"] is False


def test_mesh_e2e_states_that_no_model_produced_the_frames():
    report = benchmark_mesh_e2e.summarise(
        acks=_acks(20), rounds=20, elapsed_s=3.21,
        executor_stats={}, peer_report={})
    assert report["frames_produced_by"] == "serialiser"


# ------------------------------------------------------------------ native tcp

def test_native_tcp_is_completed_only_when_every_round_matched():
    report = benchmark_native_tcp.summarise(
        rounds=30, integrity_ok=30, refused_or_invalid=0,
        rtts=[0.9] * 30, forbidden_refused=1)
    assert report["status"] == "completed"
    assert report["acl_blocked_forbidden"] is True


def test_a_partial_native_tcp_run_is_degraded_not_completed():
    """29 of 30 byte-identical round trips is not a smaller success."""
    report = benchmark_native_tcp.summarise(
        rounds=30, integrity_ok=29, refused_or_invalid=1,
        rtts=[0.9] * 29, forbidden_refused=1)
    assert report["status"] == "degraded"


def test_a_run_with_no_rounds_at_all_is_degraded():
    report = benchmark_native_tcp.summarise(
        rounds=0, integrity_ok=0, refused_or_invalid=0,
        rtts=[], forbidden_refused=1)
    assert report["status"] == "degraded"


def test_an_acl_that_let_the_forbidden_target_through_is_reported():
    report = benchmark_native_tcp.summarise(
        rounds=30, integrity_ok=30, refused_or_invalid=0,
        rtts=[0.9] * 30, forbidden_refused=0)
    assert report["acl_blocked_forbidden"] is False


# ------------------------------------------------------------- redis cache tier

def test_the_redis_tier_percentiles_come_from_the_samples():
    """The recorded measurement is p50 0.384 / p99 0.757 ms. "0,3 ms" was
    never produced here — it is the mesh presence RTT."""
    latencies = [0.384] * 99 + [0.757]
    report = benchmark_redis_cache_tier.summarise_tier({"redis_hits": 2000},
                                                       latencies)
    assert report["hot_p50_ms"] == 0.384
    assert report["hot_p99_ms"] == 0.757
    assert report["hot_samples"] == 100


def test_an_empty_hot_phase_reports_no_latency():
    report = benchmark_redis_cache_tier.summarise_tier({}, [])
    assert report["hot_p50_ms"] is None
    assert report["hot_p99_ms"] is None
    assert report["hot_samples"] == 0
