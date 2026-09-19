import time

from neural_pods.taskgraph import TaskGraph, TaskNode


def test_parallel_speedup_over_independent_nodes():
    def slow(payload):
        time.sleep(0.4)
        return {"done": payload["index"]}

    nodes = [TaskNode(f"n{i}", lambda p, i=i: slow({"index": i})) for i in range(4)]
    graph = TaskGraph(nodes, max_parallel=4)
    result = graph.run({"index": 0})
    assert result["speedup"] > 1.5  # 4 × 0.4s sequential ≈ 1.6s; parallel < 0.8s
    assert all(r.error is None for r in result["results"].values())
    assert len(result["results"]) == 4


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
