"""Routing and statistics of the KV-cache affinity benchmark.

The HTTP pass needs vLLM replicas; everything that decides where a turn goes
and what the comparison says does not, and is checked here.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from benchmark_kvcache_affinity import Router, compare, summarize  # noqa: E402

REPLICAS = ["http://a", "http://b"]


def test_affinity_keeps_a_session_on_one_replica():
    router = Router(REPLICAS, affinity=True)
    chosen = {router.route("s1") for _ in range(10)}
    assert len(chosen) == 1
    assert router.stats()["reuses"] == 9 and router.stats()["failovers"] == 0


def test_affinity_still_spreads_distinct_sessions():
    router = Router(REPLICAS, affinity=True)
    assert {router.route(f"s{i}") for i in range(4)} == set(REPLICAS)


def test_round_robin_moves_every_turn():
    router = Router(REPLICAS, affinity=False)
    assert [router.route("s1") for _ in range(4)] == REPLICAS + REPLICAS
    assert router.stats()["reuses"] == 0


def test_empty_replica_list_is_rejected():
    with pytest.raises(ValueError, match="at least one replica"):
        Router([], affinity=True)


def test_comparison_reports_direction_and_size():
    faster = summarize("affinity", [10.0, 12.0, 11.0], Router(REPLICAS, affinity=True))
    slower = summarize("round_robin", [20.0, 22.0, 21.0], Router(REPLICAS, affinity=False))
    result = compare(faster, slower)
    assert result["affinity_helps"] is True
    assert result["ttft_p50_speedup"] == pytest.approx(21.0 / 11.0, rel=1e-3)
    assert result["ttft_p50_delta_ms"] == pytest.approx(10.0)


def test_comparison_is_undefined_without_measurements():
    empty = summarize("affinity", [], Router(REPLICAS, affinity=True))
    assert compare(empty, empty)["affinity_helps"] is None
    assert empty["ttft_p50_ms"] is None
