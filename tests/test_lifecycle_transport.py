import pytest
import torch

from neural_pods.lifecycle_transport import apply_token, compose, static_delete_token, transported_delete_token
from neural_pods.lifecycle_transport import validate_registry_binding
from neural_pods.registry import InvalidState, Registry
from neural_pods.semantic_routing import SemanticRouter
from test_semantic_routing import Encoder, index
from test_semantics import record


def rotation(seed, dim=8):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((dim, dim), generator=generator, dtype=torch.float64)
    return torch.linalg.qr(matrix).Q


def test_transport_beats_static_inverse_after_noncommutative_writes():
    writes = [(f"w{i}", rotation(i)) for i in range(5)]
    state = compose([matrix for _, matrix in writes])
    static = state @ static_delete_token(writes, "w1")
    token = transported_delete_token(writes, "w1", identity_key="K", generation_key="G7", snapshot_key="S1")
    transported = apply_token(state, token, identity_key="K", generation_key="G7", snapshot_key="S1")
    expected = compose([matrix for write_id, matrix in writes if write_id != "w1"])
    assert torch.linalg.norm(transported - expected).item() < 1e-12
    assert torch.linalg.norm(static - expected).item() > 1e-4


def test_multiple_delete_tokens_work_in_arbitrary_order_and_stale_binding_fails():
    writes = [(f"w{i}", rotation(100 + i)) for i in range(6)]
    original = compose([matrix for _, matrix in writes])
    remaining = list(writes)
    state = original
    for target_id in ["w4", "w1", "w3"]:
        token = transported_delete_token(remaining, target_id, identity_key="K", generation_key="G7", snapshot_key="S1")
        state = apply_token(state, token, identity_key="K", generation_key="G7", snapshot_key="S1")
        remaining = [(write_id, matrix) for write_id, matrix in remaining if write_id != target_id]
    expected = compose([matrix for _, matrix in remaining])
    assert torch.linalg.norm(state - expected).item() < 1e-12
    with pytest.raises(ValueError):
        apply_token(state, token, identity_key="K", generation_key="G8", snapshot_key="S1")


def test_registry_revocation_blocks_mathematically_valid_old_token(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite3")
    router = SemanticRouter(registry, Encoder(), tmp_path / "index")
    knowledge, _ = index(router, record())
    knowledge_key = knowledge["knowledge_key"]
    generation = knowledge["generation_key"]
    writes = [("w0", rotation(900)), ("w1", rotation(901))]
    token = transported_delete_token(writes, "w0", identity_key=knowledge_key,
                                     generation_key=generation, snapshot_key="S")
    validate_registry_binding(token, registry, principal="buyer")
    registry.revoke(knowledge["origin_keys"][0])
    with pytest.raises(InvalidState):
        validate_registry_binding(token, registry, principal="buyer")
    router.close(); registry.close()
