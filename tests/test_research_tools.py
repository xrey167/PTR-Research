from neural_pods.research_tools import ResearchToolCall, research_tool_manifest


def test_research_manifest_contains_typed_tools():
    names = {item["name"] for item in research_tool_manifest()}
    assert {"ann_search", "bm25_search", "graph_neighbors", "registry_snapshot"} <= names


def test_tool_call_requires_lineage_for_generation():
    ResearchToolCall("bm25_search", {"namespace": "n", "query": "q"}, namespace="n").validate()
    try:
        ResearchToolCall("ann_search", {"namespace": "n", "query": "q"}, namespace="n", generation="g1").validate()
    except ValueError as exc:
        assert "pod_identity" in str(exc)
    else:
        raise AssertionError("stale generation context was accepted without Pod identity")
