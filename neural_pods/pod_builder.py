"""Declarative Pod composition and stable-identity branching.

The builder keeps user-facing Pod identity separate from immutable artifact
revision. A branch can therefore replace generations and embeddings while a
model link continues to point at the same ``pod_identity``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

from .pod_types import PodType, typed_artifact


@dataclass(frozen=True)
class PodMetadata:
    semantic_role: str
    domain: str
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    sensitivity: str = "internal"
    confidence: float = 1.0
    acl: tuple[str, ...] = ("*",)
    temporal_scope: str | None = None
    retention: str | None = None
    cluster_key: str | None = None

    def __post_init__(self):
        if not self.semantic_role.strip() or not self.domain.strip():
            raise ValueError("semantic_role and domain are required")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")

    def as_payload(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items() if v is not None}


class PodBuilder:
    """Compose typed payloads and create immutable registry artifacts."""

    def __init__(self, registry, *, pod_type: PodType | str, name: str,
                 metadata: PodMetadata, contract: Mapping[str, Any] | None = None):
        self.registry = registry
        self.pod_type = PodType(pod_type)
        self.name = name.strip()
        if not self.name:
            raise ValueError("Pod name is required")
        self.metadata = metadata
        self.contract = dict(contract or {})
        identity_material = {"type": self.pod_type.value, "name": self.name,
                             "role": metadata.semantic_role, "domain": metadata.domain,
                             "contract": self.contract}
        self.pod_identity = "pod:" + hashlib.sha256(
            json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

    def payload(self, **fields: Any) -> dict[str, Any]:
        payload = {"semantic_type": self.pod_type.value, "pod_identity": self.pod_identity,
                   "pod_name": self.name, **self.metadata.as_payload(), **self.contract, **fields}
        return payload

    def build(self, *, parents: Sequence[str], kind: str = "model", principal: str = "local", **fields: Any) -> str:
        return typed_artifact(self.registry, kind, self.pod_type, self.payload(**fields), parents, principal)

    def branch(self, artifact_key: str, *, parents: Sequence[str], principal: str = "local", **fields: Any) -> str:
        """Create a new immutable revision retaining the same stable Pod identity."""
        node = self.registry.node(artifact_key)
        old = node["payload"].get("payload", node["payload"])
        if old.get("pod_identity") != self.pod_identity:
            raise ValueError("artifact belongs to another Pod identity")
        merged = dict(old)
        merged.update(self.payload(**fields))
        merged["branch_of"] = artifact_key
        return typed_artifact(self.registry, node["kind"], self.pod_type, merged, parents, principal)


def resolve_pod_identity(registry, pod_identity: str, *, principal: str = "local") -> str:
    """Resolve the current non-revoked artifact for a stable Pod identity."""
    rows = registry.db.execute("SELECT id FROM nodes WHERE kind IN ('model','lora','vector','text','jspace','router')").fetchall()
    candidates = []
    for row in rows:
        node = registry.node(row[0]); payload = node["payload"].get("payload", node["payload"])
        if payload.get("pod_identity") == pod_identity and not node["revoked"]:
            try:
                registry.snapshot([row[0]], principal)
                candidates.append(row[0])
            except Exception:
                continue
    if not candidates:
        raise KeyError(f"No active artifact for Pod identity {pod_identity}")
    return sorted(candidates)[-1]
