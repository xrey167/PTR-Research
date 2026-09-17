import pytest
from neural_pods.registry import Registry, InvalidState
from neural_pods.symlink import NeuralSymlinks
from neural_pods.semantic_routing import SemanticRouter
from test_semantic_routing import Encoder, index
from test_semantics import record


def setup(tmp_path):
    r = Registry(tmp_path / "r.db")
    router = SemanticRouter(r, Encoder(), tmp_path / "index")
    k, _ = index(router, record())
    links = NeuralSymlinks(r)
    links.register(k["knowledge_key"], "buyer")
    a = links.attach(k["knowledge_key"], {"fixture": True}, [("Mueller", links.target_text(k["knowledge_key"]))], "buyer")
    return r, router, links, k, a


def test_same_embedded_link_resolves_new_value_without_retraining(tmp_path):
    r, router, links, k, a = setup(tmp_path)
    text = links.target_text(k["knowledge_key"])
    before = links.resolve(text, expected_knowledge_key=k["knowledge_key"], principal="buyer")
    updated, _ = index(router, record(18, "2"))
    assert links.register(k["knowledge_key"], "buyer")["adapter_key"] == a
    after = links.resolve(text, expected_knowledge_key=k["knowledge_key"], principal="buyer")
    assert after["generation_key"] == updated["generation_key"] != before["generation_key"]
    assert after["adapter_key"] == before["adapter_key"] == a
    with pytest.raises(InvalidState): r.commit(before["snapshot"], "24 days")
    r.commit(after["snapshot"], "18 days")
    router.close(); r.close()


@pytest.mark.parametrize("change", ["aliases", "cluster"])
def test_changed_identity_or_cluster_blocks_old_link(tmp_path, change):
    r, router, links, k, _ = setup(tmp_path)
    h = record(18, "2")
    if change == "aliases": h["trusted_aliases"] = []
    else: h["domain"] = "another-domain"
    index(router, h)
    with pytest.raises(InvalidState, match="changed"):
        links.resolve(links.target_text(k["knowledge_key"]), expected_knowledge_key=k["knowledge_key"], principal="buyer")
    links.register(k["knowledge_key"], "buyer")
    with pytest.raises(InvalidState, match="no trained"):
        links.active_binding(k["knowledge_key"], "buyer")
    router.close(); r.close()


@pytest.mark.parametrize("prediction", ["LINK 999 CLUSTER 1", "LINK 1 CLUSTER 999", "24 days", "LINK 1 CLUSTER 1 extra"])
def test_wrong_or_malformed_model_pointer_is_rejected(tmp_path, prediction):
    r, router, links, k, _ = setup(tmp_path)
    with pytest.raises(InvalidState): links.resolve(prediction, expected_knowledge_key=k["knowledge_key"], principal="buyer")
    router.close(); r.close()


def test_cluster_is_shared_but_source_revocation_is_selective(tmp_path):
    r, router, links, k, a = setup(tmp_path)
    h = record()
    h.update(subject_id="supplier:other", subject_label="Other", trusted_aliases=[])
    h["source"]["record_id"] = "other"
    other, _ = index(router, h)
    links.register(other["knowledge_key"], "buyer")
    links.attach(other["knowledge_key"], {}, [("Other", links.target_text(other["knowledge_key"]))], "buyer")
    first = links.active_binding(k["knowledge_key"], "buyer")
    assert k["semantic_cluster"]["key"] == first["cluster_key"]
    assert len(links.cluster_members(first["cluster_key"], "buyer")) == 2
    assert links.active_binding(other["knowledge_key"], "buyer")["cluster_key"] == first["cluster_key"]
    pending = links.resolve(links.target_text(k["knowledge_key"]), expected_knowledge_key=k["knowledge_key"], principal="buyer")["snapshot"]
    assert a in r.revoke(k["origin_keys"][0])
    with pytest.raises(InvalidState): r.commit(pending, "24 days")
    assert links.cluster_members(first["cluster_key"], "buyer") == [other["knowledge_key"]]
    assert links.cluster_members(first["cluster_key"], "outsider") == []
    router.close(); r.close()
