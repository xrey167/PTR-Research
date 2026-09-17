"""Immutable execution handoff for serving runtimes.

The manifest is the narrow boundary between lifecycle-aware retrieval and a
model server (for example vLLM-Omni). A server may schedule the listed stages,
but it cannot reinterpret lineage or silently switch generations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .registry import InvalidState, Snapshot


@dataclass(frozen=True)
class ExecutionManifest:
    generation_key: str
    artifact_keys: tuple[str, ...]
    origin_keys: tuple[str, ...]
    reader_key: str | None
    snapshot: Snapshot

    @classmethod
    def build(cls, registry, generation_key: str, artifact_keys: Iterable[str],
              *, reader_key: str | None = None, principal: str = "local"):
        artifacts = tuple(sorted(set(artifact_keys)))
        if not artifacts:
            raise InvalidState("Execution manifest needs at least one artifact")
        deps = [generation_key, *artifacts]
        if reader_key:
            deps.append(reader_key)
        snapshot = registry.snapshot(deps, principal)
        origins = set(registry.roots(generation_key))
        generation = registry.node(generation_key)
        if generation["kind"] != "knowledge":
            raise InvalidState("Manifest generation must be a knowledge node")
        for artifact in artifacts:
            node = registry.node(artifact)
            if generation_key not in node["payload"].get("generations", {}):
                raise InvalidState("Artifact is not derived from the requested generation")
            origins.update(node["payload"].get("origin_keys", []))
        if reader_key:
            reader = registry.node(reader_key)
            if reader["kind"] not in {"lora", "model"}:
                raise InvalidState("Reader binding must be a lora or model artifact")
            # A reader adapter is executable model state.  Bind it to the
            # exact knowledge generation carried by this manifest so a
            # value-bearing adapter cannot silently answer from another
            # generation (or from an unrelated corpus).
            if generation_key not in reader["payload"].get("generations", {}):
                raise InvalidState("Reader binding is not derived from the requested generation")
        return cls(generation_key, artifacts, tuple(sorted(origins)), reader_key, snapshot)

    def validate(self, registry):
        """Re-check the manifest at the execution/commit boundary."""
        deps = [self.generation_key, *self.artifact_keys]
        if self.reader_key:
            deps.append(self.reader_key)
        current = registry.snapshot(deps, self.snapshot.principal)
        if current.artifacts != self.snapshot.artifacts:
            raise InvalidState("Execution manifest dependency set changed")
        return current

    def as_dict(self):
        return {"generation_key": self.generation_key,
                "artifact_keys": list(self.artifact_keys),
                "origin_keys": list(self.origin_keys),
                "reader_key": self.reader_key,
                "principal": self.snapshot.principal}
