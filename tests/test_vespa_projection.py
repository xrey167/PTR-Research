from neural_pods import LocalSearchBackend, TensorRankProfile, onnx_ranker


def test_partial_update_is_realtime_and_revision_checked(tmp_path):
    backend = LocalSearchBackend(tmp_path / "search.sqlite3")
    backend.create_namespace("pods")
    backend.upsert("pods", "a", text="Muller lead time", metadata={"status": "active", "generation_key": "g7"})
    assert backend.search("pods", text="Muller")[0].metadata["generation_key"] == "g7"
    assert backend.update_metadata("pods", "a", {"generation_key": "g8"}, expected_revision=1) == 2
    assert backend.search("pods", text="Muller")[0].metadata["generation_key"] == "g8"
    try:
        backend.update_metadata("pods", "a", {"generation_key": "g9"}, expected_revision=1)
    except ValueError as exc:
        assert "revision conflict" in str(exc)
    else:
        raise AssertionError("stale partial update was accepted")
    backend.close()


def test_tensor_rank_profile_runs_after_hard_filters(tmp_path):
    backend = LocalSearchBackend(tmp_path / "search.sqlite3")
    backend.create_namespace("pods")
    backend.upsert("pods", "public", text="supplier risk", vector=[1.0, 0.0],
                   metadata={"status": "active", "risk": 0.1})
    backend.upsert("pods", "preferred", text="supplier risk", vector=[0.9, 0.1],
                   metadata={"status": "active", "risk": 0.9})
    profile = TensorRankProfile(vector_weight=0.0, lexical_weight=0.0,
                                regex_weight=0.0, metadata_weights={"risk": 1.0})
    hits = backend.search("pods", text="supplier", vector=[1.0, 0.0],
                          filters={"status": "active"}, rank_profile=profile)
    assert [hit.key for hit in hits] == ["preferred", "public"]
    backend.close()


def test_onnx_adapter_contract_without_loading_a_model():
    class FakeSession:
        def run(self, _outputs, inputs):
            assert list(inputs) == ["features"]
            return [[[sum(inputs["features"][0])]]]

    scorer = onnx_ranker(FakeSession())
    assert scorer((1.0, 2.0, 3.0), {}) == 6.0
