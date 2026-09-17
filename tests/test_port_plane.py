import pytest

from neural_pods.registry import InvalidState, Registry
from neural_pods.symlink import TemporalPortPlane
from neural_pods.semantic_routing import SemanticRouter
from test_semantic_routing import Encoder, index
from test_semantics import record


def setup(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite3")
    router = SemanticRouter(registry, Encoder(), tmp_path / "index")
    knowledge, _ = index(router, record(24, "1"))
    plane = TemporalPortPlane(registry)
    return registry, router, plane, knowledge


def test_aliases_and_values_are_late_bound_without_training(tmp_path):
    registry, router, plane, knowledge = setup(tmp_path)
    key = knowledge["knowledge_key"]
    bound = plane.bind(key, ["Müller delivery", "supplier muller"], value_handle="value:24d", principal="buyer")
    assert bound["revision"] == 1
    first = plane.resolve("MULLER DELIVERY", principal="buyer")
    assert first["value_handle"] == "value:24d"
    updated, _ = index(router, record(18, "2"))
    changed = plane.update_value(key, "value:18d", expected_revision=1, principal="buyer")
    second = plane.resolve("supplier muller", principal="buyer")
    assert changed["revision"] == 2
    assert second["generation_key"] == updated["generation_key"]
    assert second["value_handle"] == "value:18d"
    with pytest.raises(InvalidState):
        plane.update_value(key, "value:stale", expected_revision=1, principal="buyer")
    router.close(); registry.close()


def test_revocation_blocks_old_port_snapshot_and_alias_conflicts(tmp_path):
    registry, router, plane, knowledge = setup(tmp_path)
    key = knowledge["knowledge_key"]
    plane.bind(key, ["Müller"], value_handle="value:24d", principal="buyer")
    pending = plane.resolve("Müller", principal="buyer")["snapshot"]
    other_hard = record(12, "3")
    other_hard.update(subject_id="supplier:other", subject_label="Other", trusted_aliases=["Other supplier"])
    other_hard["source"]["record_id"] = "other"
    other, _ = index(router, other_hard)
    other_key = other["knowledge_key"]
    with pytest.raises(InvalidState):
        plane.bind(other_key, ["Müller"], value_handle="value:12d", principal="buyer")
    registry.revoke(knowledge["origin_keys"][0])
    with pytest.raises(InvalidState):
        registry.commit(pending, "24 days")
    with pytest.raises(InvalidState):
        plane.resolve("Müller", principal="buyer")
    router.close(); registry.close()
