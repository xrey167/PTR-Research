from copy import deepcopy
from datetime import datetime, timezone, timedelta
import pytest
from neural_pods.registry import Registry, InvalidState
from neural_pods.semantics import SemanticCompiler, SemanticRole, resolve_query


def record(value=24, version="1"):
    return {"subject_id": "supplier:muller", "subject_label": "Müller GmbH", "predicate": "lead_time",
            "component_id": "x12", "component_label": "X12", "object": {"value": value, "unit": "days"},
            "role": "FACT", "type": "supplier_metric", "owner": "procurement", "acl": ["buyer"],
            "trusted_aliases": ["Müller", "Muller GmbH", "Mueller", "Mueller GmbH"],
            "trusted_tags": ["supplier", "lead-time"], "domain": "procurement", "language": "de",
            "source": {"system": "sap", "record_id": "92831", "version": version,
                       "content": {"lead_time_days": value}}}


@pytest.mark.parametrize("question", [
    "What is the delivery lead time for X99 from Mueller GmbH?",
    "Mueller delivery lead time for X12 and X99?",
    "Mueller delivery lead time for unknown-widget?",
    "How long has Mueller GmbH existed?",
    "How long are Mueller GmbH payment terms for X12?",
    "Wie lange existiert Mueller GmbH?",
])
def test_unresolved_query_constraints_are_rejected(tmp_path, question):
    r = Registry(tmp_path / "r.db")
    k = SemanticCompiler(r).compile(record(), principal="buyer")
    with pytest.raises(InvalidState):
        resolve_query(question, [r.node(k["generation_key"])["payload"]["semantic"]])
    r.close()


@pytest.mark.parametrize("left,right", [
    (("supplier:acme:x12", "y55", "lead_time", "FACT"), ("supplier:acme", "x12:y55", "lead_time", "FACT")),
    (("supplier:acme", None, "description:rule", "FACT"), ("supplier:acme", None, "description", "RULE")),
    (("supplier:acme:x12", None, "description", "FACT"), ("supplier:acme", "x12", "description", "FACT")),
])
def test_distinct_structured_identities_do_not_supersede(tmp_path, left, right):
    r = Registry(tmp_path / "r.db")
    c = SemanticCompiler(r)
    keys = []
    for subject, component, predicate, role in (left, right):
        hard = record()
        hard.update(subject_id=subject, component_id=component, predicate=predicate, role=role)
        keys.append(c.compile(hard, principal="buyer"))
    assert keys[0]["knowledge_key"] != keys[1]["knowledge_key"]
    for k in keys: r.snapshot([k["generation_key"]], principal="buyer")
    r.close()


def test_legacy_registry_rejects_mixed_identity_writes(tmp_path):
    r = Registry(tmp_path / "r.db")
    origin = r.origin("test", "1", "1", {}, acl=["buyer"])
    r.publish("knowledge:legacy", {"transform_chain": ["semantic-compiler:v1"]}, [origin], principal="buyer", acl=["buyer"])
    before = r.db.execute("SELECT count(*) FROM nodes").fetchone()[0]
    with pytest.raises(InvalidState, match="fresh registry"):
        SemanticCompiler(r).compile(record(), principal="buyer")
    assert r.db.execute("SELECT count(*) FROM nodes").fetchone()[0] == before
    r.close()


def test_aliases_share_one_identity_and_one_lifecycle(tmp_path):
    r = Registry(tmp_path / "r.db")
    c = SemanticCompiler(r)
    first = c.compile(record(), principal="buyer")
    assert first["knowledge_key"].startswith("knowledge:v2:")
    semantic = r.node(first["generation_key"])["payload"]["semantic"]
    for question in ["Current delivery lead time for supplier Müller?", "X12 procurement delay from Müller GmbH?",
                     "How long does Mueller need for X12?", "Wie lange braucht Muller GmbH fuer X12?",
                     "How many days does Müller GmbH need to deliver X12?"]:
        assert resolve_query(question, [semantic])["subject"] == "supplier:muller"
    pod = r.artifact("lora", {}, [first["generation_key"]], principal="buyer")
    snap = r.snapshot([pod], principal="buyer")
    second = c.compile(record(18, "2"), principal="buyer")
    assert second["knowledge_key"] == first["knowledge_key"]
    assert second["generation"] == first["generation"] + 1
    assert second["generation_key"] != first["generation_key"]
    assert second["origin_keys"] != first["origin_keys"]
    assert second["lifecycle"]["supersedes"] == first["generation_key"]
    with pytest.raises(InvalidState): r.commit(snap, "24 days")
    r.close()


def test_unrelated_company_question_does_not_activate_lead_time(tmp_path):
    r = Registry(tmp_path / "r.db")
    cko = SemanticCompiler(r).compile(record(), principal="buyer")
    s = r.node(cko["generation_key"])["payload"]["semantic"]
    with pytest.raises(InvalidState, match="intent"):
        resolve_query("Who founded Müller GmbH?", [s])
    r.close()


@pytest.mark.parametrize("field", ["acl", "owner", "generation", "subject_id", "valid_until", "source"])
def test_soft_annotation_cannot_override_authority(tmp_path, field):
    r = Registry(tmp_path / "r.db")
    with pytest.raises(ValueError):
        SemanticCompiler(r).compile(record(), [{"field": field, "value": "fake", "source": "llm", "confidence": 1.0}], principal="buyer")
    assert r.db.execute("SELECT count(*) FROM nodes").fetchone()[0] == 0
    r.close()


def test_soft_alias_advisory_not_canonical_identity(tmp_path):
    r = Registry(tmp_path / "r.db")
    x = SemanticCompiler(r).compile(record(), [{"field": "aliases", "value": ["Unrelated Company"], "source": "llm", "confidence": 0.8}], principal="buyer")
    s = r.node(x["generation_key"])["payload"]["semantic"]
    assert x["retrieval"]["soft_annotations"][0]["confidence"] == 0.8
    with pytest.raises(InvalidState): resolve_query("Delivery time from Unrelated Company?", [s])
    r.close()


def test_expiry_during_inference_and_valid_from_boundary(tmp_path):
    now = [datetime(2026, 9, 15, 12, tzinfo=timezone.utc)]
    r = Registry(tmp_path / "r.db", clock=lambda: now[0])
    hard = record()
    hard.update(valid_from=now[0].isoformat(), valid_until=(now[0]+timedelta(seconds=10)).isoformat())
    k = SemanticCompiler(r).compile(hard, principal="buyer")["generation_key"]
    pod = r.artifact("lora", {}, [k], principal="buyer")
    snap = r.snapshot([pod], principal="buyer")
    now[0] += timedelta(seconds=10)
    with pytest.raises(InvalidState, match="expired"): r.commit(snap, "24 days")
    now[0] -= timedelta(seconds=11)
    with pytest.raises(InvalidState, match="not yet"): r.snapshot([pod], principal="buyer")
    r.close()


@pytest.mark.parametrize("start,end", [("2026-09-15T12:00:00", None),
    ("2026-09-15T12:00:00+00:00", "2026-09-15T11:00:00+00:00")])
def test_invalid_time_rejected_before_writes(tmp_path, start, end):
    r = Registry(tmp_path / "r.db")
    hard = record()
    hard.update(valid_from=start, valid_until=end)
    with pytest.raises(ValueError): SemanticCompiler(r).compile(hard, principal="buyer")
    assert r.db.execute("SELECT count(*) FROM nodes").fetchone()[0] == 0
    r.close()


@pytest.mark.parametrize("role", list(SemanticRole))
def test_explicit_semantic_roles_preserved(tmp_path, role):
    r = Registry(tmp_path / "r.db")
    hard = record()
    hard.update(role=role.value, predicate="description", object="typed payload")
    x = SemanticCompiler(r).compile(hard, principal="buyer")
    assert x["knowledge"]["role"] == role.value
    r.close()


def test_longer_company_name_does_not_match_shorter_alias(tmp_path):
    r = Registry(tmp_path / "r.db")
    compiler = SemanticCompiler(r)
    a = compiler.compile(record(), principal="buyer")
    hard = record()
    hard.update(subject_id="supplier:muller-werke", subject_label="Müller Werke", trusted_aliases=[])
    b = compiler.compile(hard, principal="buyer")
    semantics = [r.node(x["generation_key"])["payload"]["semantic"] for x in [a, b]]
    assert resolve_query("Müller Werke delivery time for X12?", semantics)["subject"] == "supplier:muller-werke"
    r.close()
