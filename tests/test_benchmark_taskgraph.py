"""The task-graph verdict, checked against known observation sets.

This is the benchmark whose reported "speedup 2.59x" was a metric artefact:
the four "parallel" nodes ran one after another on the paho loop thread, and
the number rose precisely BECAUSE they queued. It survived because it was
computed inside a function that needed an MQTT broker to run, so nothing
could feed it a case with a known answer.

`summarise()` is now pure. These tests feed it both a parallel run and a
serialised one and require the verdict to FLIP. A test that only shows the
green case proves that the arithmetic runs, not that it discriminates.
"""
import pytest

from research.benchmark_taskgraph import REMOTE_DELAY_S, summarise

NODES = ("ingest_a", "ingest_b", "ingest_c", "ingest_d", "plan", "verify")


def _run(wall_s, *, outputs=None, errors=None):
    """One TaskGraph run as the benchmark sees it, as plain dicts."""
    outputs = outputs or {}
    errors = errors or {}
    return {
        "wall_s": wall_s,
        "node_elapsed_sum_s": round(len(NODES) * REMOTE_DELAY_S, 3),
        "critical_path_s": round(2 * REMOTE_DELAY_S, 3),
        "mean_concurrency": round(len(NODES) * REMOTE_DELAY_S / wall_s, 2),
        "critical_path_ratio": round(2 * REMOTE_DELAY_S / wall_s, 2),
        "results": {
            node: {"node_id": node,
                   "output": outputs.get(node, {"echo": node,
                                                "processed_by": "tg-responder"}),
                   "error": errors.get(node),
                   "started": 0.0, "finished": REMOTE_DELAY_S,
                   "inputs_from": ()}
            for node in NODES},
    }


def test_a_parallel_run_lands_within_the_dag_bound():
    """Two levels deep at 100 ms each: real parallelism finishes near 0.2 s."""
    report = summarise(_run(0.24))
    assert report["dag_delay_bound_s"] == 0.2
    assert report["wall_within_bound"] is True
    assert report["correct"] is True
    assert report["nodes"] == 6


def test_a_serialised_run_misses_the_bound():
    """Six nodes answered one at a time need 6 x 100 ms. This is the case the
    old ratio could not distinguish — and the case that actually happened."""
    report = summarise(_run(0.61))
    assert report["wall_within_bound"] is False


def test_the_bound_is_exactly_twice_the_dag_depth_delay():
    """The threshold is 2x the bound, so 0.4 s passes and 0.41 s does not.
    Pinned because an off-by-one factor here silently widens the check."""
    assert summarise(_run(0.40))["wall_within_bound"] is True
    assert summarise(_run(0.41))["wall_within_bound"] is False


def test_mean_concurrency_rises_with_queueing_and_is_therefore_not_the_verdict():
    """The property that made the old metric useless, pinned so nobody
    reinstates it: the SLOWER run reports the HIGHER concurrency, because
    node durations include the time spent queueing."""
    fast, slow = _run(0.24), _run(0.61)
    # Same work, same node durations, different wall time.
    assert fast["node_elapsed_sum_s"] == slow["node_elapsed_sum_s"]
    assert fast["mean_concurrency"] > slow["mean_concurrency"]
    # And the verdict does not come from that number.
    assert summarise(fast)["wall_within_bound"] is True
    assert summarise(slow)["wall_within_bound"] is False


def test_a_node_that_raised_makes_the_run_incorrect():
    report = summarise(_run(0.24, errors={"plan": "RuntimeError('boom')"}))
    assert report["correct"] is False


def test_a_timeout_reported_inside_the_output_is_not_a_success():
    """`remote_call` returns {"error": "timeout"} instead of raising, so a
    check that only reads the node's `error` field counts it as correct."""
    report = summarise(_run(0.24, outputs={"verify": {"error": "timeout"}}))
    assert report["correct"] is False


def test_a_node_with_no_output_at_all_is_not_a_success():
    report = summarise(_run(0.24, outputs={"ingest_a": None}))
    assert report["correct"] is False


def test_the_raw_observations_survive_into_the_report():
    """The gate reads `results` to count nodes; dropping or reshaping them
    here would make the count silently wrong."""
    report = summarise(_run(0.24))
    assert set(report["results"]) == set(NODES)
    assert report["results"]["plan"]["output"]["echo"] == "plan"
    assert report["mean_concurrency"] == _run(0.24)["mean_concurrency"]


def test_summarise_accepts_the_dataclasses_a_live_run_produces():
    """A summary that only accepts plain dicts would work in these tests and
    fail on the server, which is the opposite of useful."""
    from neural_pods.taskgraph import TaskResult

    run = _run(0.24)
    run["results"] = {
        node: TaskResult(node_id=node, output={"echo": node}, error=None,
                         started=0.0, finished=REMOTE_DELAY_S)
        for node in NODES}
    report = summarise(run)
    assert report["correct"] is True
    assert report["results"]["plan"]["output"] == {"echo": "plan"}
