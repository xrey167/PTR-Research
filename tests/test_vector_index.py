from neural_pods.vector_index import PersistentVectorIndex
import pytest


def test_persistent_vector_index_roundtrip(tmp_path):
    path = tmp_path / "vectors"
    idx = PersistentVectorIndex(path)
    idx.upsert("a", [1, 0]); idx.upsert("b", [0, 1]); idx.save()
    restored = PersistentVectorIndex(path)
    assert restored.search([.9, .1], 1)[0][0] == "a"
    assert restored.stats()["exact"] is True


def test_hnsw_index_optional_acceleration(tmp_path):
    try:
        from neural_pods.vector_index import HNSWVectorIndex
        idx = HNSWVectorIndex(tmp_path / "hnsw", dimension=2)
    except RuntimeError:
        pytest.skip("hnswlib unavailable")
    idx.upsert("a", [1, 0]); idx.upsert("b", [0, 1]); idx.save()
    assert idx.search([.9, .1], 1)[0][0] == "a"
    assert idx.stats()["storage"] == "hnswlib"
    restored = HNSWVectorIndex(tmp_path / "hnsw", dimension=2)
    assert restored.search([.9, .1], 1)[0][0] == "a"
