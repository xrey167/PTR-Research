from datetime import datetime, timezone, timedelta
import numpy as np
import pytest
from neural_pods.registry import Registry, InvalidState
from neural_pods.semantics import SemanticCompiler
from neural_pods.semantic_routing import SemanticRouter
from test_semantics import record


class Encoder:
    def encode(self, text, normalize_embeddings=True):
        return np.array([1.0, 0.0], dtype=np.float32)


def index(router, hard, soft=()):
    k = SemanticCompiler(router.registry).compile(hard, soft, principal="buyer")
    a = router.registry.artifact("text", {"fixture": True}, [k["generation_key"]], principal="buyer")
    router.index(k["generation_key"], a, principal="buyer")
    return k, a


@pytest.mark.parametrize("invalidity", ["superseded", "revoked", "expired", "private"])
def test_invalid_alias_cannot_resolve_or_conflict_with_reassignment(tmp_path, invalidity):
    now = [datetime(2026, 9, 15, 12, tzinfo=timezone.utc)]
    r = Registry(tmp_path / "r.db", clock=lambda: now[0])
    router = SemanticRouter(r, Encoder(), tmp_path / "index")
    old = record()
    old.update(subject_id="supplier:alpha", subject_label="Alpha Supply", trusted_aliases=["Shared Co"])
    if invalidity == "expired": old["valid_until"] = (now[0] + timedelta(seconds=1)).isoformat()
    if invalidity == "private": old["acl"] = ["buyer"]
    first, _ = index(router, old)
    principal = "outsider" if invalidity == "private" else "buyer"
    if invalidity == "superseded":
        replacement = record(18, "2")
        replacement.update(subject_id="supplier:alpha", subject_label="Alpha Supply", trusted_aliases=[])
        index(router, replacement)
    elif invalidity == "revoked": r.revoke(first["origin_keys"][0])
    elif invalidity == "expired": now[0] += timedelta(seconds=1)
    question = "Shared Co delivery lead time for X12?"
    with pytest.raises(InvalidState): router.select(question, principal=principal, learned=False)
    new = record(30, "3")
    new.update(subject_id="supplier:beta", subject_label="Beta Supply", trusted_aliases=["Shared Co"], acl=["*"])
    second, _ = index(router, new)
    assert router.select(question, principal=principal, learned=False)["knowledge_key"] == second["knowledge_key"]
    router.close()
    r.close()


@pytest.mark.parametrize("question", [
    "What is the delivery lead time for X99 from Mueller GmbH?",
    "Mueller delivery lead time for X12 and X99?",
    "How long has Mueller GmbH existed?",
])
def test_bad_questions_do_not_reach_materialization(tmp_path, question):
    r = Registry(tmp_path / "r.db")
    router = SemanticRouter(r, Encoder(), tmp_path / "index")
    index(router, record())
    with pytest.raises(InvalidState):
        selection = router.select(question, principal="buyer", learned=False)
        r.commit(selection["snapshot"], "24 days")
    assert r.db.execute("SELECT count(*) FROM nodes WHERE kind='answer'").fetchone()[0] == 0
    router.close()
    r.close()


def test_filters_apply_before_ann_and_registry_is_authority(tmp_path):
    r = Registry(tmp_path / "r.db")
    router = SemanticRouter(r, Encoder(), tmp_path / "index")
    k, _ = index(router, record())
    wrong = record(99)
    wrong.update(subject_id="supplier:other", subject_label="Other Company", trusted_aliases=[])
    index(router, wrong)
    # Identical vectors make semantic constraints necessary to get the right entity.
    chosen = router.select("Current delivery lead time for supplier Müller?", principal="buyer", learned=False,
                           filters={"tags": ["lead-time"], "domain": "procurement", "language": "de", "type": "supplier_metric"})
    assert chosen["knowledge_key"] == k["knowledge_key"]
    assert chosen["ann_eligible_count"] == 1
    filter_keys = {x["key"] for x in chosen["qdrant_filter"]["must"]}
    assert {"generation_key", "subject", "predicate", "role", "status", "acl", "valid_from", "valid_until", "tags", "domain", "language", "type"} <= filter_keys
    router.client.set_payload("semantic_pods", {"evidence": "999 days", "adapter": "foreign"}, points=[router.bindings()[0]["point_id"]])
    chosen = router.select("X12 procurement delay from Müller GmbH?", principal="buyer", learned=False)
    assert "24 days" in chosen["evidence"]
    router.close()
    r.close()


def test_acl_generation_time_and_soft_tags_do_not_escape_filters(tmp_path):
    now = [datetime(2026, 9, 15, 12, tzinfo=timezone.utc)]
    r = Registry(tmp_path / "r.db", clock=lambda: now[0])
    router = SemanticRouter(r, Encoder(), tmp_path / "index")
    hard = record()
    hard["valid_until"] = (now[0]+timedelta(hours=1)).isoformat()
    k, _ = index(router, hard, [{"field": "tags", "value": ["secret"], "source": "llm", "confidence": 0.99}])
    q = "Müller delivery lead time for X12?"
    with pytest.raises(InvalidState): router.select(q, principal="outsider", learned=False)
    with pytest.raises(InvalidState): router.select(q, principal="buyer", learned=False, filters={"tags": ["secret"]})
    selected = router.select(q, principal="buyer", learned=False)
    now[0] += timedelta(hours=1)
    with pytest.raises(InvalidState): r.commit(selected["snapshot"], "24 days")
    with pytest.raises(InvalidState): router.select(q, principal="buyer", learned=False)
    updated, _ = index(router, record(18, "2"))
    selected = router.select(q, principal="buyer", learned=False)
    assert selected["generation_key"] == updated["generation_key"]
    assert "18 days" in selected["evidence"]
    r.revoke(updated["origin_keys"][0])
    with pytest.raises(InvalidState): router.select(q, principal="buyer", learned=False)
    router.close()
    r.close()


def test_multiple_components_require_disambiguation(tmp_path):
    r = Registry(tmp_path / "r.db")
    router = SemanticRouter(r, Encoder(), tmp_path / "index")
    index(router, record())
    other = record(40)
    other.update(component_id="y55", component_label="Y55")
    index(router, other)
    with pytest.raises(InvalidState, match="ambiguous"):
        router.select("Current delivery lead time for supplier Müller?", principal="buyer", learned=False)
    assert "24 days" in router.select("Muller delivery lead time for X12?", principal="buyer", learned=False)["evidence"]
    router.close()
    r.close()


def test_role_prevents_rule_from_becoming_a_fact(tmp_path):
    r = Registry(tmp_path / "r.db")
    router = SemanticRouter(r, Encoder(), tmp_path / "index")
    k, _ = index(router, record())
    rule = record(99)
    rule["role"] = "RULE"
    other, _ = index(router, rule)
    assert k["knowledge_key"] != other["knowledge_key"]
    assert router.select("Müller delivery time for X12?", principal="buyer", learned=False)["knowledge_key"] == k["knowledge_key"]
    router.close()
    r.close()


def test_relation_walk_pins_all_visited_generations(tmp_path):
    r = Registry(tmp_path / "r.db")
    compiler = SemanticCompiler(r)
    target = compiler.compile(record(), principal="buyer")
    hard = record()
    hard.update(predicate="supplier_profile", role="ORGANIZATION", object="supplier profile",
                relations=[{"predicate": "has_metric", "target_knowledge_key": target["knowledge_key"]}])
    parent = compiler.compile(hard, principal="buyer")
    walk = compiler.walk_relations(parent["knowledge_key"], ["has_metric"], principal="buyer")
    assert len(walk["generations"]) == 2
    compiler.compile(record(18, "2"), principal="buyer")
    with pytest.raises(InvalidState): r.commit(walk["snapshot"], "old derived answer")
    r.close()


def test_learned_scorer_reload_and_private_training_lineage(tmp_path):
    r = Registry(tmp_path / "r.db")
    router = SemanticRouter(r, Encoder(), tmp_path / "index")
    k, _ = index(router, record())
    router.fit([("Müller delivery lead time for X12?", k["knowledge_key"])], principal="buyer")
    assert router.select("X12 procurement delay from Müller GmbH?", principal="buyer")["knowledge_key"] == k["knowledge_key"]
    with pytest.raises(InvalidState): r.snapshot([router.router_artifact], principal="outsider")
    router.close()
    reloaded = SemanticRouter(r, Encoder(), tmp_path / "index")
    reloaded.load_weights()
    assert reloaded.select("How many days does Muller need to deliver X12?", principal="buyer")["knowledge_key"] == k["knowledge_key"]
    reloaded.close()
    r.close()
