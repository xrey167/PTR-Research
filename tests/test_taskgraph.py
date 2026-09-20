import threading
import time

from neural_pods.taskgraph import TaskGraph, TaskNode


def test_independent_nodes_run_concurrently():
    """Four independent 0.4 s sleeps: wall stays near one node's duration, so
    mean_concurrency approaches 4 and the critical path (one node) is close to
    the wall time."""
    def slow(payload):
        time.sleep(0.4)
        return {"done": payload["index"]}

    nodes = [TaskNode(f"n{i}", lambda p, i=i: slow({"index": i})) for i in range(4)]
    graph = TaskGraph(nodes, max_parallel=4)
    result = graph.run({"index": 0})
    assert result["wall_s"] < 0.8                 # not 4 × 0.4 s
    assert result["mean_concurrency"] > 3.0       # ~4 nodes in flight
    assert result["critical_path_ratio"] > 0.8    # scheduler at the DAG bound
    assert all(r.error is None for r in result["results"].values())
    assert len(result["results"]) == 4


def test_mean_concurrency_is_not_reported_as_speedup():
    """A serialising resource inflates node durations, so mean_concurrency
    rises while nothing actually ran in parallel. The metric must not be
    called a speedup, and critical_path_ratio must stay near 1.0 — the chain
    of measured durations IS the wall time here."""
    gate = threading.Lock()

    def serialised(_payload):
        with gate:                                 # one at a time
            time.sleep(0.2)
        return "ok"

    nodes = [TaskNode(f"n{i}", serialised) for i in range(4)]
    result = TaskGraph(nodes, max_parallel=4).run()
    assert "speedup" not in result
    assert result["mean_concurrency"] > 1.5        # inflated by queueing alone
    assert result["wall_s"] >= 0.8                 # 4 × 0.2 s, strictly serial
    assert result["critical_path_ratio"] > 0.8


def test_critical_path_follows_the_dependency_chain():
    def slow(_payload):
        time.sleep(0.2)
        return "ok"

    graph = TaskGraph([
        TaskNode("a", slow),
        TaskNode("b", slow, depends_on=("a",)),
        TaskNode("c", slow),                       # independent of the chain
    ], max_parallel=3)
    result = graph.run()
    # a -> b is the longest chain: ~0.4 s, not the 0.6 s of all three nodes.
    assert 0.35 < result["critical_path_s"] < 0.55
    assert result["node_elapsed_sum_s"] > result["critical_path_s"]


def test_token_flows_from_dependency_into_payload():
    captured = {}

    def producer(payload):
        return {"tokens": "answer: 52 days"}

    def consumer(payload):
        captured["context"] = payload["context"]
        return {"echo": payload["context"]["tokens"]}

    graph = TaskGraph([
        TaskNode("produce", producer),
        TaskNode("consume", consumer, depends_on=("produce",)),
    ])
    result = graph.run()
    assert captured["context"] == {"tokens": "answer: 52 days"}
    assert result["results"]["consume"].output["echo"] == "answer: 52 days"
    assert result["results"]["consume"].inputs_from == ("produce",)


def test_multiple_inputs_merged_as_dict():
    def a(_p):
        return "A"
    def b(_p):
        return "B"
    def merge(payload):
        return {"a": payload["context"]["a"], "b": payload["context"]["b"]}

    graph = TaskGraph([
        TaskNode("a", a), TaskNode("b", b),
        TaskNode("merge", merge, depends_on=("a", "b"), context_key="context"),
    ])
    graph.run()
    outputs = graph.outputs()
    assert outputs["merge"] == {"a": "A", "b": "B"}


def test_node_error_is_recorded_not_raised():
    def boom(_p):
        raise RuntimeError("pod down")
    graph = TaskGraph([TaskNode("boom", boom)])
    result = graph.run()
    assert "pod down" in result["results"]["boom"].error


def test_cycles_and_unknown_deps_rejected():
    try:
        TaskGraph([
            TaskNode("a", lambda p: p, depends_on=("b",)),
            TaskNode("b", lambda p: p, depends_on=("a",)),
        ])
        raise AssertionError("expected cycle rejection")
    except ValueError as error:
        assert "cycle" in str(error)
    try:
        TaskGraph([TaskNode("a", lambda p: p, depends_on=("ghost",))])
        raise AssertionError("expected unknown-dependency rejection")
    except ValueError as error:
        assert "unknown dependency" in str(error)
