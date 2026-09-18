"""Typed semantic compiler. Hard fields enter through a trusted structured record.

Soft enrichment can influence discovery text, never identity, ACL or validity.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
import re
import unicodedata
from .registry import InvalidState, digest


class SemanticRole(str, Enum):
    FACT = "FACT"
    RULE = "RULE"
    EVENT = "EVENT"
    PROCEDURE = "PROCEDURE"
    PREFERENCE = "PREFERENCE"
    CONSTRAINT = "CONSTRAINT"
    DEFINITION = "DEFINITION"
    RELATION = "RELATION"
    TIME_SERIES = "TIME_SERIES"
    DOCUMENT = "DOCUMENT"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PRODUCT = "PRODUCT"
    LOCATION = "LOCATION"


def normalized(text):
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                            if not unicodedata.combining(c)).split())


def utc_stamp(value):
    if value is None: return None
    date = datetime.fromisoformat(value)
    if date.tzinfo is None: raise ValueError("Timestamp requires timezone")
    return date.astimezone(timezone.utc).isoformat()


def cluster_descriptor(semantic):
    """Versioned grouping from hard semantics, independent of values and aliases."""
    return {"schema": "semantic-cluster:v1", "domain": semantic["retrieval"]["domain"],
            "type": semantic["type"], "predicate": semantic["predicate"], "role": semantic["role"]}


def semantic_cluster(semantic):
    descriptor = cluster_descriptor(semantic)
    return {"key": "cluster:v1:" + digest(descriptor), "descriptor": descriptor}


class SemanticCompiler:
    VERSION = "semantic-compiler:v2"
    SOFT_FIELDS = {"tags", "topics", "aliases", "relations", "summary", "classification"}

    def __init__(self, registry):
        self.registry = registry

    def compile(self, hard, soft=(), principal="local"):
        h = deepcopy(hard)
        required = {"subject_id", "subject_label", "predicate", "object", "source", "owner", "acl", "role"}
        if not required <= h.keys(): raise ValueError(f"Missing hard fields: {sorted(required - h.keys())}")
        role = SemanticRole(h["role"]).value
        for field in ("subject_id", "predicate"):
            if not isinstance(h[field], str) or not re.fullmatch(r"[a-z0-9:_-]+", h[field]):
                raise ValueError(f"Invalid canonical {field}")
        if not isinstance(h["acl"], list) or not h["acl"] or not all(isinstance(x, str) and x for x in h["acl"]):
            raise ValueError("An explicit nonempty ACL is required")
        if not isinstance(h["owner"], str) or not h["owner"]:
            raise ValueError("Owner is required")
        component = h.get("component_id")
        if component is not None and not re.fullmatch(r"[a-z0-9:_-]+", component):
            raise ValueError("Invalid canonical component_id")
        if h["predicate"] == "lead_time":
            obj = h["object"]
            if not isinstance(obj, dict) or obj.get("unit") != "days" or type(obj.get("value")) not in (int, float) or obj["value"] < 0:
                raise ValueError("lead_time requires a nonnegative numeric value in days")
            if not component or not h.get("component_label"):
                raise ValueError("lead_time requires an explicit component")
        source = h["source"]
        if not {"system", "record_id", "version", "content"} <= source.keys():
            raise ValueError("Source identity, version and raw content are required")
        start, end = utc_stamp(h.get("valid_from")), utc_stamp(h.get("valid_until"))
        if start and end and start >= end: raise ValueError("Invalid validity interval")
        enrichment = []
        for annotation in soft:
            a = deepcopy(annotation)
            if set(a) != {"field", "value", "source", "confidence"} or a["field"] not in self.SOFT_FIELDS:
                raise ValueError("Soft enrichment cannot set hard metadata")
            if type(a["confidence"]) not in (float, int) or not 0 <= a["confidence"] <= 1 or not a["source"]:
                raise ValueError("Soft annotations require source and confidence in [0,1]")
            enrichment.append(a)
        # Explicit canonical IDs never come from the display name or an LLM alias.
        key = "knowledge:v2:" + digest([h["subject_id"], component, h["predicate"], role])
        # Historical immutable runs remain readable, but must not acquire a
        # second identity for the same fact through a mixed-schema write.
        for row in self.registry.db.execute("SELECT node_id FROM heads"):
            payload = self.registry.node(row[0])["payload"]
            semantic = payload.get("semantic", {})
            if "semantic-compiler:v1" in semantic.get("transform_chain", []):
                raise InvalidState("Legacy semantic identities: rebuild in a fresh registry before compiling v2")
        retrieval = {"trusted_aliases": sorted(set([h["subject_label"], *h.get("trusted_aliases", [])])),
                     "entities": [h["subject_id"]] + ([component] if component else []),
                     "tags": sorted(set(h.get("trusted_tags", []))), "domain": h.get("domain", "general"),
                     "language": h.get("language", "und"), "soft_annotations": enrichment}
        knowledge = {"role": role, "type": h.get("type", role.lower()), "subject": h["subject_id"],
                     "subject_label": h["subject_label"], "predicate": h["predicate"], "object": h["object"],
                     "component": component, "component_label": h.get("component_label"),
                     "relations": h.get("relations", []), "constraints": h.get("constraints", []),
                     "uncertainty": h.get("uncertainty"), "retrieval": retrieval,
                     "transform_chain": ["normalize:unicode-nfkd-v1", self.VERSION]}
        # Validate serializability/nonfinite numbers before creating any nodes.
        digest({"knowledge": knowledge, "hard": h})
        origin = self.registry.origin(source["system"], source["record_id"], source["version"], source["content"], acl=h["acl"])
        parents = [origin, *h.get("depends_on", [])]
        node = self.registry.publish(key, knowledge, parents, principal=principal, acl=h["acl"],
                    lifecycle={"owner": h["owner"], "valid_from": start, "valid_until": end,
                               "depends_on": h.get("depends_on", []), "source": origin})
        return self.describe(node)

    def describe(self, node):
        p = self.registry.node(node)["payload"]
        return {"origin_keys": self.registry.roots(node), "knowledge_key": p["knowledge_key"],
                "generation_key": node, "generation": p["generation"],
                "semantic_cluster": semantic_cluster(p["semantic"]),
                "retrieval": p["semantic"]["retrieval"],
                "knowledge": {k: v for k, v in p["semantic"].items() if k != "retrieval"},
                "lifecycle": {**p["lifecycle"], "acl": p["acl"], "revoked": bool(self.registry.node(node)["revoked"])}}

    def walk_relations(self, knowledge_key, predicates, max_hops=2, principal="local"):
        """Resolve canonical relation targets at current generations and pin all reads."""
        if not 0 <= max_hops <= 16: raise ValueError("max_hops must be between 0 and 16")
        queue, visited, edges = [(knowledge_key, 0)], {}, []
        while queue:
            key, depth = queue.pop(0)
            if key in visited: continue
            node = self.registry.head(key)
            self.registry.snapshot([node], principal)
            visited[key] = node
            if len(visited) > 1000: raise InvalidState("Relation traversal exceeds the node limit")
            if depth >= max_hops: continue
            for relation in self.registry.node(node)["payload"]["semantic"].get("relations", []):
                if relation.get("predicate") in predicates and relation.get("target_knowledge_key"):
                    target = relation["target_knowledge_key"]
                    edges.append({"from": key, "predicate": relation["predicate"], "to": target})
                    queue.append((target, depth + 1))
        snapshot = self.registry.snapshot(visited.values(), principal)
        return {"generations": visited, "edges": edges, "snapshot": snapshot}


def resolve_query(question, semantics):
    """Resolve explicit known names; soft aliases cannot establish an identity."""
    q = normalized(question)
    matches = []
    for semantic in semantics:
        for alias in semantic["retrieval"]["trusted_aliases"]:
            for match in re.finditer(r"(?<!\w)" + re.escape(normalized(alias)) + r"(?!\w)", q):
                matches.append((match.start(), match.end(), semantic["subject"]))
    # A longer known name disambiguates its shorter prefix (Müller vs Müller Werke).
    matches = [m for m in matches if not any(a <= m[0] and b >= m[1] and b-a > m[1]-m[0] for a,b,_ in matches)]
    subjects = sorted({m[2] for m in matches})
    component_matches = [(m.start(), m.end(), s["component"]) for s in semantics if s.get("component_label")
                         for m in re.finditer(r"(?<!\w)" + re.escape(normalized(s["component_label"])) + r"(?!\w)", q)]
    components = sorted({m[2] for m in component_matches})
    intent = bool(re.search(r"\b(lead[ -]time|procurement delay|delivery|deliver|lieferzeit|lieferdauer|liefertage)\b", q))
    intent = intent or bool(components and re.search(r"\b(how long.*need|wie lange.*braucht)\b", q))
    if not intent: raise InvalidState("Unsupported or unclear query intent")
    if len(subjects) != 1: raise InvalidState("Supplier is missing or ambiguous")
    if len(components) > 1: raise InvalidState("Multiple components require explicit multi-hop planning")
    # This prototype supports a bounded metric-query vocabulary. Reject any
    # unresolved content instead of interpreting an unknown part as omission.
    remainder = list(q)
    for start, end, _ in matches + component_matches:
        remainder[start:end] = " " * (end - start)
    vocabulary = set("what is the current recorded delivery lead time procurement delay for supplier from how long does need to deliver many days state of give in supplied by should i plan and tell me duration component part wie lange braucht welche lieferzeit hat fur fuer lieferdauer liefertage die der von was ist dauert lieferung bitte a an".split())
    unknown = set(re.findall(r"\w+", "".join(remainder))) - vocabulary
    if unknown: raise InvalidState("Unresolved component or unsupported query content")
    return {"subject": subjects[0], "component": components[0] if components else None,
            "predicate": "lead_time", "role": "FACT", "intent": "supplier_metric_lookup"}


def evidence_text(semantic):
    obj = semantic["object"]
    return f"The delivery lead time for {semantic['component_label']} from {semantic['subject_label']} is {obj['value']} {obj['unit']}."
