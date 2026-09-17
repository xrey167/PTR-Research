from neural_pods.local_search import LocalSearchBackend
from neural_pods.recursive_search import RecursiveSearchRunner


def test_recursive_runner_deepens_incomplete_broad_search_and_consolidates(tmp_path):
    backend = LocalSearchBackend(tmp_path / "search.sqlite")
    backend.create_namespace("corpus")
    backend.upsert("corpus", "a", text="Phil Redmond created Brookside", vector=[1, 0], metadata={"status": "active"})
    backend.upsert("corpus", "b", text="Brookside premiered on Channel 4", vector=[.9, .1], metadata={"status": "active"})
    backend.upsert("corpus", "noise", text="unrelated document", vector=[0, 1], metadata={"status": "active"})
    result = RecursiveSearchRunner(backend, "corpus", max_deep_turns=2).run(
        "Which documents explain Brookside and Channel 4?", ["a", "b"])
    assert result.broad.recall < 1.0
    assert result.deep is not None
    assert result.deep.recall == 1.0
    assert result.pods
    assert any(x.verified for x in result.experiences)
    backend.close()
