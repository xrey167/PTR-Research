"""Structured Pod type/domain/tag routing metadata."""
from __future__ import annotations
from dataclasses import dataclass, field


POD_TYPES = {"context", "math", "model", "reasoning", "retrieval"}

# These are intentionally hard tags. Soft LLM annotations may add tags, but
# cannot replace the minimum contract required for a training example.
REQUIRED_TRAINING_TAGS = {
    "context": frozenset({"pod:context", "role:knowledge"}),
    "math": frozenset({"pod:math", "role:calculation", "has:unit"}),
    "model": frozenset({"pod:model", "role:execution", "has:interface"}),
    "reasoning": frozenset({"pod:reasoning", "role:inference"}),
    "retrieval": frozenset({"pod:retrieval", "role:search"}),
}
REQUIRED_TRAINING_FIELDS = frozenset({
    "id", "pod_type", "domain", "semantic_role", "tags", "origin_keys",
    "knowledge_key", "generation_key", "split", "input", "target",
})


def validate_training_record(record):
    """Validate the hard metadata contract before a row enters training.

    ``tags`` can contain arbitrary advisory labels, but the type-specific
    required tags and lifecycle keys are mandatory. Returns the record to make
    it convenient to use in dataset pipelines.
    """
    missing = REQUIRED_TRAINING_FIELDS - set(record)
    if missing:
        raise ValueError(f"training record missing fields: {sorted(missing)}")
    kind = record["pod_type"]
    if kind not in POD_TYPES:
        raise ValueError(f"unknown pod_type: {kind}")
    if not isinstance(record["tags"], (list, tuple, set)):
        raise ValueError("training record tags must be a sequence")
    missing_tags = REQUIRED_TRAINING_TAGS[kind] - set(record["tags"])
    if missing_tags:
        raise ValueError(f"{kind} training record missing required tags: {sorted(missing_tags)}")
    if not record["origin_keys"] or not record["knowledge_key"] or not record["generation_key"]:
        raise ValueError("training record needs origin, knowledge and generation lineage")
    if record["split"] not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    return record


@dataclass(frozen=True)
class PodDescriptor:
    artifact_key: str
    pod_type: str
    domain: str
    tags: frozenset[str] = field(default_factory=frozenset)
    hard: bool = True

    def __post_init__(self):
        if self.pod_type not in POD_TYPES:
            raise ValueError(f"unknown pod type: {self.pod_type}")
        if not self.domain.strip():
            raise ValueError("domain is required")


def route_candidates(pods, *, domain=None, pod_type=None, required_tags=(),
                     soft_domain=None):
    """Apply hard taxonomy filters, then score advisory domain/tag overlap.

    Lifecycle, ACL and generation checks happen outside this pure metadata
    function. Vela predictions belong in ``soft_domain`` and cannot override
    hard Pod type or lifecycle constraints.
    """
    required = set(required_tags)
    candidates = [p for p in pods if (domain is None or p.domain == domain)
                  and (pod_type is None or p.pod_type == pod_type)
                  and required.issubset(p.tags)]
    soft_domain = set(soft_domain or ())
    def score(p):
        return (2.0 if p.domain in soft_domain else 0.0) + len(required & p.tags)
    return sorted(candidates, key=score, reverse=True)
