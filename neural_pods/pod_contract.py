"""Shared Pod, link and activation contracts.

This module is deliberately transport/model agnostic. It is the stable semantic
boundary used by local, SSH, WebSocket and future mTLS Pod runtimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
from typing import Iterable, Mapping

from .registry import InvalidState


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class PodManifest:
    pod_id: str
    generation: str
    artifact_id: str
    pod_type: str
    model_track: str
    model_family: str
    interface: str
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    namespace: str = "default"
    acl: tuple[str, ...] = ("*",)
    status: str = "candidate"
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.pod_id or not self.generation or not self.artifact_id:
            raise ValueError("Pod identity, generation and artifact_id are required")
        if self.status not in {"candidate", "trained", "evaluated", "approved", "active", "superseded", "retired", "revoked"}:
            raise ValueError(f"invalid lifecycle status: {self.status}")
        if not self.capabilities:
            raise ValueError("Pod manifest needs at least one capability")
        if "origin_key" not in self.provenance:
            raise ValueError("Pod manifest requires provenance.origin_key")

    def as_dict(self):
        result = asdict(self)
        for key in ("capabilities", "tags", "acl"):
            result[key] = list(result[key])
        return result

    @property
    def manifest_hash(self) -> str:
        return _hash(self.as_dict())


@dataclass(frozen=True)
class PodLink:
    trace_id: str
    source_pod_id: str
    target_pod_id: str
    target_generation: str
    artifact_id: str
    transport: str
    capability: str
    acl: tuple[str, ...]
    deadline_ms: int = 1500
    hop_budget: int = 3
    visited: tuple[str, ...] = ()
    attestation: str = ""

    def validate(self, target: PodManifest, *, principal: str, now=None):
        if self.target_pod_id != target.pod_id or self.target_generation != target.generation:
            raise InvalidState("Pod link targets a stale or different generation")
        if self.artifact_id != target.artifact_id:
            raise InvalidState("Pod link artifact does not match active manifest")
        if self.capability not in target.capabilities:
            raise InvalidState("Target Pod does not advertise requested capability")
        if target.status != "active":
            raise InvalidState("Target Pod is not active")
        if target.acl != ("*",) and principal not in target.acl:
            raise InvalidState("Pod link denied by target ACL")
        if self.acl != ("*",) and principal not in self.acl:
            raise InvalidState("Pod link denied by link ACL")
        if self.deadline_ms < 1 or self.hop_budget < 1:
            raise InvalidState("Pod link deadline and hop budget must be positive")
        if self.target_pod_id in self.visited:
            raise InvalidState("Pod link would create a cycle")
        if self.hop_budget <= len(self.visited):
            raise InvalidState("Pod link hop budget exhausted")
        expected = target.manifest_hash
        if self.attestation and self.attestation != expected:
            raise InvalidState("Pod attestation does not match target manifest")
        return True

    def next_hop(self) -> "PodLink":
        return PodLink(self.trace_id, self.source_pod_id, self.target_pod_id,
                       self.target_generation, self.artifact_id, self.transport,
                       self.capability, self.acl, self.deadline_ms,
                       self.hop_budget - 1, (*self.visited, self.source_pod_id), self.attestation)


class LifecycleGate:
    """Activation gate for immutable manifests and registry artifacts."""

    ALLOWED = {"candidate": {"trained"}, "trained": {"evaluated"},
               "evaluated": {"approved", "candidate"}, "approved": {"active"},
               "active": {"superseded", "revoked", "retired"}}

    @classmethod
    def transition(cls, manifest: PodManifest, status: str, *, checks: Mapping[str, bool]) -> PodManifest:
        if status not in cls.ALLOWED.get(manifest.status, set()):
            raise InvalidState(f"illegal lifecycle transition {manifest.status}->{status}")
        required = {"manifest_hash", "provenance", "base_identity"}
        if status in {"evaluated", "approved", "active"} and not required <= {k for k, v in checks.items() if v}:
            raise InvalidState("activation requires manifest_hash, provenance and base_identity checks")
        return PodManifest(**{**manifest.as_dict(), "status": status})
