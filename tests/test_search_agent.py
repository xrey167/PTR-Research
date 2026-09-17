from neural_pods.local_search import LocalSearchBackend
from neural_pods.search_agent import LocalSearchAgent, SearchAction


def test_iterative_agent_fans_out_tools_and_measures_recall(tmp_path):
    backend = LocalSearchBackend(tmp_path / "search.sqlite")
    backend.create_namespace("corpus")
    backend.upsert("corpus", "doc-a", text="Phil Redmond created Brookside", vector=[1, 0], metadata={"status": "active"})
    backend.upsert("corpus", "doc-b", text="Brookside premiered on Channel 4", vector=[.9, .1], metadata={"status": "active"})
    backend.upsert("corpus", "noise", text="unrelated document", vector=[0, 1], metadata={"status": "active"})
    agent = LocalSearchAgent(backend, "corpus", max_turns=2, parallelism=4)

    def policy(question, turn, observed):
        if turn == 0:
            return [SearchAction("bm25", text="Phil Redmond"), SearchAction("regex", regex=r"Channel 4")]
        return [SearchAction("bm25", text="Brookside premiered")]

    episode = agent.run("Which documents explain the answer?", ["doc-a", "doc-b"], policy)
    assert episode.recall == 1.0
    assert episode.search_calls == 2
    assert episode.reward > 0.8
    assert {h.key for h in episode.hits} >= {"doc-a", "doc-b"}
    backend.close()
