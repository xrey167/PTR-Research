import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pytest
import torch
from neural_pods.dragonfly import DragonflyRouter
from neural_pods.registry import Registry, InvalidState
from test_semantic_routing import index
from test_semantics import record


class Encoder:
    def encode(self, text, normalize_embeddings=True):
        if "Alpha" in text: return np.array([1., 0., 0.], dtype=np.float32)
        if "Beta" in text: return np.array([0., 1., 0.], dtype=np.float32)
        return np.array([0., 0., 1.], dtype=np.float32)


def setup_pods(tmp_path, clock=None, expires=None):
    r = Registry(tmp_path / "r.db", clock=clock)
    router = DragonflyRouter(r, Encoder(), tmp_path / "index", encoder_id="test-encoder:v1")
    pods = []
    for name in ("Alpha", "Beta"):
        h = record()
        h.update(subject_id="supplier:" + name.lower(), subject_label=name, trusted_aliases=[])
        h["source"]["record_id"] = name
        if expires: h["valid_until"] = expires
        pods.append(index(router, h)[0])
    for i, name in enumerate(("Alpha", "Beta")):
        router.fit_pod(pods[i]["generation_key"], [f"{name} delivery time for X12?"],
                       [f"{'Beta' if i == 0 else 'Alpha'} delivery time for X12?"], principal="buyer", steps=100)
    return r, router, pods


def test_learned_z_changes_scores_and_survives_reload(tmp_path):
    r, router, pods = setup_pods(tmp_path)
    selected = router.select("Alpha delivery lead time for X12?", principal="buyer")
    assert selected["knowledge_key"] == pods[0]["knowledge_key"]
    rep = selected["representation_key"]
    assert rep in selected["snapshot"].artifacts
    p = r.node(rep)["payload"]["payload"]
    assert p["z_delta_l2"] > 0.1 and p["loss_after"] < p["loss_before"]
    item = next(i for i in router.bindings() if i["generation_key"] == pods[0]["generation_key"])
    # Hold metadata constant: discrimination must come from the neural address.
    positive = router.learned_score(Encoder().encode("Alpha"), [1.] * 9, item)
    negative = router.learned_score(Encoder().encode("Beta"), [1.] * 9, item)
    assert positive > 0.9 and negative < 0.1
    router.close()
    reloaded = DragonflyRouter(r, Encoder(), tmp_path / "index", encoder_id="test-encoder:v1")
    reloaded.load_weights()
    again = reloaded.select("Alpha delivery lead time for X12?", principal="buyer")
    assert again["score"] == selected["score"] and again["representation_key"] == rep
    reloaded.close()
    r.close()


@pytest.mark.parametrize("target", ["source", "generation", "representation", "training_origin"])
def test_revocation_is_selective_and_blocks_pending_commit(tmp_path, target):
    r, router, pods = setup_pods(tmp_path)
    chosen = router.select("Alpha delivery time for X12?", principal="buyer")
    rep = chosen["representation_key"]
    training = next(n["id"] for n in r.ancestors(rep) if n["kind"] == "origin" and n["payload"]["namespace"] == "dragonfly-address-training")
    target_id = {"source": pods[0]["origin_keys"][0], "generation": pods[0]["generation_key"],
                 "representation": rep, "training_origin": training}[target]
    assert rep in r.revoke(target_id)
    with pytest.raises(InvalidState): r.commit(chosen["snapshot"], "24 days")
    with pytest.raises(InvalidState): router.select("Alpha delivery time for X12?", principal="buyer")
    assert router.select("Beta delivery time for X12?", principal="buyer")["knowledge_key"] == pods[1]["knowledge_key"]
    router.close()
    r.close()


def test_update_needs_new_representation_and_never_reuses_old_z(tmp_path):
    r, router, pods = setup_pods(tmp_path)
    before = router.select("Alpha delivery time for X12?", principal="buyer")
    h = record(18, "2")
    h.update(subject_id="supplier:alpha", subject_label="Alpha", trusted_aliases=[])
    updated, _ = index(router, h)
    with pytest.raises(InvalidState): router.select("Alpha delivery time for X12?", principal="buyer")
    router.fit_pod(updated["generation_key"], ["Alpha delivery time for X12?"], ["Beta delivery time for X12?"], principal="buyer", steps=100)
    after = router.select("Alpha delivery time for X12?", principal="buyer")
    assert after["representation_key"] != before["representation_key"]
    assert after["generation_key"] == updated["generation_key"]
    with pytest.raises(InvalidState): r.commit(before["snapshot"], "24 days")
    router.close()
    r.close()


@pytest.mark.parametrize("question,principal", [
    ("Alpha delivery time for X99?", "buyer"),
    ("How long has Alpha existed?", "buyer"),
    ("Alpha delivery time for X12?", "outsider"),
])
def test_learned_score_does_not_bypass_hard_gates(tmp_path, question, principal):
    r, router, _ = setup_pods(tmp_path)
    for _, model in router.addresses.values():
        with torch.no_grad(): model.bias.fill_(-1000)
    with pytest.raises(InvalidState): router.select(question, principal=principal)
    router.close()
    r.close()


def test_representation_tampering_and_encoder_mismatch_rejected(tmp_path):
    r, router, _ = setup_pods(tmp_path)
    router.encoder_id = "another-encoder"
    with pytest.raises(InvalidState, match="Incompatible"): router.load_weights()
    router.encoder_id = "test-encoder:v1"
    key = next(iter(router.addresses.values()))[0]
    payload = r.node(key)["payload"]
    payload["payload"]["z"][0] += 1
    r.db.execute("UPDATE nodes SET payload=? WHERE id=?", (json.dumps(payload), key))
    with pytest.raises(InvalidState, match="hash"): router.load_weights()
    router.close()
    r.close()


def test_expiry_blocks_loaded_representation_and_inflight_answer(tmp_path):
    now = [datetime(2026, 9, 15, 12, tzinfo=timezone.utc)]
    r, router, _ = setup_pods(tmp_path, clock=lambda: now[0], expires=(now[0] + timedelta(seconds=1)).isoformat())
    selection = router.select("Alpha delivery time for X12?", principal="buyer")
    now[0] += timedelta(seconds=1)
    with pytest.raises(InvalidState): router.select("Alpha delivery time for X12?", principal="buyer")
    with pytest.raises(InvalidState): r.commit(selection["snapshot"], "24 days")
    router.close()
    r.close()


def test_derived_training_evidence_remains_in_lineage(tmp_path):
    r, router, pods = setup_pods(tmp_path)
    source = r.origin("analyst", "routing-examples", "1", {"fixture": True}, acl=["buyer"])
    trained = router.fit_pod(pods[0]["generation_key"], ["Alpha delivery time for X12?"],
                            ["Beta delivery time for X12?"], principal="buyer", training_parents=[source], steps=100)
    assert source in r.roots(trained["representation_key"])
    r.revoke(source)
    with pytest.raises(InvalidState): router.select("Alpha delivery time for X12?", principal="buyer")
    assert router.select("Beta delivery time for X12?", principal="buyer")["knowledge_key"] == pods[1]["knowledge_key"]
    router.close()
    r.close()
