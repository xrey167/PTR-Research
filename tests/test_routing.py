"""Real local Qdrant with fixed vectors isolates the index/registry boundary."""
import numpy as np
import torch
from qdrant_client import QdrantClient, models
from neural_pods.data import FACTS
from neural_pods.registry import Registry, InvalidState
from neural_pods.routing import Router


class FixedEncoder:
    def encode(self, question, normalize_embeddings=True):
        return np.array([1.0, 0.0], dtype=np.float32)


def setup_router(tmp_path):
    r = Registry(tmp_path / "registry.db")
    fact = FACTS[0]
    o = r.origin("fixture", fact.key, "1", fact.semantic())
    k = r.publish(fact.key, fact.semantic(), [o])
    pod = r.artifact("lora", {"adapter": "correct_adapter"}, [k])
    vector = r.artifact("vector", {"embedding": [1.0, 0.0]}, [k])
    router = Router.__new__(Router)
    router.registry = r
    router.keys = [fact.key]
    router.facts = {fact.key: fact}
    router.encoder = FixedEncoder()
    router.net = torch.nn.Linear(2, 1)
    router.client = QdrantClient(":memory:")
    router.client.create_collection("pods", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    router.client.upsert("pods", [models.PointStruct(id=1, vector=[1.0, 0.0], payload={
        "key": fact.key, "generation": 999, "adapter": "wrong_adapter", "evidence": "999 days",
        "vector_artifact": vector, "pod_artifact": pod})])
    return router, o, k


def test_index_metadata_cannot_override_canonical_payload(tmp_path):
    router, _, _ = setup_router(tmp_path)
    selected = router.select("Lead time for X12?", learned=False)
    assert selected["generation"] == 1
    assert selected["adapter"] == "correct_adapter"
    assert selected["evidence"] == FACTS[0].evidence
    router.close()
    router.registry.close()


def test_stale_index_result_rejected_after_origin_revoke(tmp_path):
    router, origin, _ = setup_router(tmp_path)
    router.registry.revoke(origin)
    assert router.candidates("Lead time for X12?") == []
    router.close()
    router.registry.close()
