import numpy as np
import pytest
from neural_pods.dragonfly import DragonflyRouter, alias_training_questions
from neural_pods.registry import Registry, InvalidState
from test_semantics import record
from test_semantic_routing import index


class AliasEncoder:
    def encode(self, text, normalize_embeddings=True):
        # Unrelated vectors for aliases: the mapping must be learned.
        for position, name in enumerate(["alpha", "beta", "northstar", "southbank"]):
            if name in text.casefold(): return np.eye(4, dtype=np.float32)[position]
        return np.zeros(4, dtype=np.float32)


def test_alias_is_learned_automatically_and_available_after_activation(tmp_path, monkeypatch):
    r = Registry(tmp_path / "r.db")
    router = DragonflyRouter(r, AliasEncoder(), tmp_path / "index", encoder_id="aliases:v1")
    for name, alias, other, other_alias in [("Alpha", "Northstar", "Beta", "Southbank"),
                                          ("Beta", "Southbank", "Alpha", "Northstar")]:
        h = record()
        h.update(subject_id="supplier:" + name.lower(), subject_label=name, trusted_aliases=[alias])
        h["source"]["record_id"] = name
        k, _ = index(router, h)
        # No positive question contains the alias: fit_pod must add it itself.
        result = router.fit_pod(k["generation_key"], [f"{name} delivery time for X12?"],
                                [f"{other} delivery time for X12?"], negative_aliases=[other_alias, other], principal="buyer")
        assert alias in result["alias_training"]["aliases"]
    router.close()
    loaded = DragonflyRouter(r, AliasEncoder(), tmp_path / "index", encoder_id="aliases:v1")
    loaded.load_weights()
    # Disallow any access to the old lexical name resolver for the diagnostic.
    def forbidden(*args, **kwargs): raise AssertionError("Lexical resolver used")
    monkeypatch.setattr("neural_pods.semantic_routing.resolve_query", forbidden)
    for alias, expected in [("Northstar", "supplier:alpha"), ("Southbank", "supplier:beta")]:
        predicted = loaded.recognize_alias(alias, principal="buyer")
        assert predicted["subject"] == expected and predicted["score"] >= 0.8
        assert predicted["representation_key"] in predicted["snapshot"].artifacts
    with pytest.raises(InvalidState): loaded.recognize_alias("Unknown Company", principal="buyer")
    with pytest.raises(InvalidState): loaded.recognize_alias("Northstar", principal="outsider")
    alpha = loaded.recognize_alias("Northstar", principal="buyer")
    r.revoke(alpha["representation_key"])
    with pytest.raises(InvalidState): loaded.recognize_alias("Northstar", principal="buyer")
    assert loaded.recognize_alias("Southbank", principal="buyer")["subject"] == "supplier:beta"
    loaded.close()
    r.close()


def test_soft_aliases_do_not_enter_automatic_training():
    semantic = {"retrieval": {"trusted_aliases": ["Alpha", "A-Team"],
                "soft_annotations": [{"field": "aliases", "value": ["Ghost"]}]}}
    aliases, questions = alias_training_questions(semantic, ["Alpha delivery time for X12?"])
    assert "A-Team delivery time for X12?" in questions
    assert "A-Team" in questions and "a-team" in questions
    assert not any("Ghost" in q for q in questions) and "Ghost" not in aliases


def test_alias_removal_requires_current_generation_and_retraining(tmp_path):
    r = Registry(tmp_path / "r.db")
    router = DragonflyRouter(r, AliasEncoder(), tmp_path / "index", encoder_id="aliases:v1")
    h = record()
    h.update(subject_id="supplier:alpha", subject_label="Alpha", trusted_aliases=["Northstar"])
    first, _ = index(router, h)
    router.fit_pod(first["generation_key"], ["Alpha delivery time for X12?"], ["Beta"], principal="buyer")
    assert router.recognize_alias("Northstar", principal="buyer")["subject"] == "supplier:alpha"
    h["trusted_aliases"] = []
    h["source"]["version"] = "2"
    second, _ = index(router, h)
    with pytest.raises(InvalidState): router.recognize_alias("Northstar", principal="buyer")
    router.fit_pod(second["generation_key"], ["Alpha delivery time for X12?"], ["Beta"],
                   negative_aliases=["Northstar"], principal="buyer")
    with pytest.raises(InvalidState): router.recognize_alias("Northstar", principal="buyer")
    assert router.recognize_alias("Alpha", principal="buyer")["subject"] == "supplier:alpha"
    router.close()
    r.close()


def test_insufficient_alias_learning_does_not_activate_artifact(tmp_path):
    from test_semantic_routing import Encoder
    r = Registry(tmp_path / "r.db")
    router = DragonflyRouter(r, Encoder(), tmp_path / "index", encoder_id="constant-test:v1")
    k, _ = index(router, record())
    before = r.db.execute("SELECT count(*) FROM nodes").fetchone()[0]
    # Constant vectors make positive/negative aliases impossible to distinguish.
    with pytest.raises(InvalidState, match="coverage"):
        router.fit_pod(k["generation_key"], ["Mueller delivery time for X12?"], ["Other Company"], principal="buyer")
    assert not router.addresses
    assert r.db.execute("SELECT count(*) FROM nodes").fetchone()[0] == before
    router.close()
    r.close()
