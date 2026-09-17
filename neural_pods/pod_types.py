"""Typed Pod contracts used by retrieval, model and math memory paths.

Pods share one lineage envelope, but their payload contracts differ. Keeping
the type explicit prevents a math fact from being treated as free-form context
or an adapter artifact from being executed as a fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping
import hashlib
import json


class PodType(StrEnum):
    CONTEXT = "context"
    MODEL = "model"
    MATH = "math"
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    ROUTER = "router"
    TRANSPORT = "transport"
    RUNTIME = "runtime"
    VISION = "vision"
    POLICY = "policy"
    EMBEDDING = "embedding"
    QUANTIZATION = "quantization"


class ModelVariant(StrEnum):
    """Executable model payload variants carried by a model Pod."""
    LORA = "lora"
    DISTILLED = "distilled"
    MOE = "moe"
    QUANTIZED = "quantized"
    BASE = "base"
    SPECULATOR = "speculator"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class PodHeader:
    pod_id: str
    pod_type: PodType
    origin_keys: tuple[str, ...]
    knowledge_keys: tuple[str, ...]
    generation_keys: tuple[str, ...]
    artifact_key: str
    tags: tuple[str, ...] = ()
    surprise_gate: Mapping[str, Any] | None = None
    model_variant: str | None = None
    model_family: str | None = None
    post_training: str | None = None
    interface: str | None = None
    # Stable across immutable branches; artifact_key identifies one revision.
    pod_identity: str | None = None
    semantic_role: str | None = None
    domain: str | None = None
    aliases: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    sensitivity: str | None = None
    confidence: float | None = None

    @classmethod
    def from_artifact(cls, artifact_key: str, node: Mapping[str, Any]) -> "PodHeader":
        envelope = node.get("payload", {})
        payload = envelope.get("payload", envelope)
        kind = payload.get("semantic_type", payload.get("pod_type"))
        try:
            pod_type = PodType(kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("Pod artifact needs semantic_type=context|model|math") from exc
        generations = envelope.get("generations", {})
        if not generations:
            raise ValueError("Pod artifact must bind at least one knowledge generation")
        generation_keys = tuple(sorted(generations))
        knowledge_keys = tuple(sorted({generations[key]["knowledge_key"] for key in generation_keys}))
        return cls(pod_id=node["id"], pod_type=pod_type,
                   origin_keys=tuple(envelope.get("origin_keys", ())),
                   knowledge_keys=knowledge_keys, generation_keys=generation_keys,
                   artifact_key=artifact_key, tags=tuple(payload.get("tags", ())),
                   surprise_gate=payload.get("surprise_gate"),
                   model_variant=payload.get("model_variant"),
                   model_family=payload.get("model_family"),
                   post_training=payload.get("post_training"),
                   interface=payload.get("interface"), pod_identity=payload.get("pod_identity"),
                   semantic_role=payload.get("semantic_role"), domain=payload.get("domain"),
                   aliases=tuple(payload.get("aliases", ())), capabilities=tuple(payload.get("capabilities", ())),
                   entities=tuple(payload.get("entity_ids", payload.get("entities", ()))),
                   sensitivity=payload.get("sensitivity"), confidence=payload.get("confidence"))

    @property
    def knowledge_key(self) -> str | None:
        """Compatibility alias; multi-source Pods intentionally return None."""
        return self.knowledge_keys[0] if len(self.knowledge_keys) == 1 else None

    @property
    def generation_key(self) -> str | None:
        """Compatibility alias; multi-generation Pods intentionally return None."""
        return self.generation_keys[0] if len(self.generation_keys) == 1 else None


def validate_payload(pod_type: PodType | str, payload: Mapping[str, Any]) -> None:
    """Validate type-specific hard fields before a Pod enters a model path."""
    kind = PodType(pod_type)
    if kind is PodType.CONTEXT:
        if not payload.get("content") and not payload.get("text"):
            raise ValueError("context Pod requires content or text")
        gate = payload.get("surprise_gate", {})
        if gate and not {"novelty_threshold", "contradiction_threshold"} <= set(gate):
            raise ValueError("surprise_gate needs novelty and contradiction thresholds")
    elif kind is PodType.MODEL:
        for key in ("reader_identity", "input_schema", "output_schema"):
            if not payload.get(key):
                raise ValueError(f"model Pod requires {key}")
        variant = payload.get("model_variant", ModelVariant.LORA.value)
        if "model_family" in payload and not isinstance(payload["model_family"], str):
            raise ValueError("model_family must be a string when supplied")
        for key in ("post_training", "interface"):
            if key in payload and not isinstance(payload[key], str):
                raise ValueError(f"{key} must be a string when supplied")
        for key in ("router_ref", "ranker_ref"):
            if key in payload and payload[key] is not None and not isinstance(payload[key], str):
                raise ValueError(f"{key} must be a string when supplied")
        if "stage_role" in payload and payload["stage_role"] is not None and not isinstance(payload["stage_role"], str):
            raise ValueError("stage_role must be a string when supplied")
        for key in ("parallel_strategy", "cache_policy"):
            if key in payload and payload[key] is not None and not isinstance(payload[key], Mapping):
                raise ValueError(f"{key} must be a mapping when supplied")
        try:
            ModelVariant(variant)
        except ValueError as exc:
            raise ValueError(f"unknown model Pod variant: {variant}") from exc
        if not payload.get("adapter_sha256") and not payload.get("model_sha256"):
            raise ValueError("model Pod requires adapter_sha256 or model_sha256")
        if variant == ModelVariant.MOE.value and not payload.get("experts"):
            raise ValueError("MoE model Pod requires experts metadata")
        if variant == ModelVariant.DISTILLED.value and not payload.get("teacher_identity"):
            raise ValueError("distilled model Pod requires teacher_identity")
    elif kind is PodType.MATH:
        for key in ("expression", "operands", "unit"):
            if key not in payload:
                raise ValueError(f"math Pod requires {key}")
        if not isinstance(payload["operands"], (list, tuple, dict)):
            raise ValueError("math Pod operands must be structured")
    elif kind in {PodType.REASONING, PodType.RETRIEVAL, PodType.ROUTER, PodType.TRANSPORT,
                  PodType.RUNTIME, PodType.VISION, PodType.POLICY, PodType.EMBEDDING,
                  PodType.QUANTIZATION}:
        if not payload.get("content") and not payload.get("text") and not payload.get("capabilities"):
            raise ValueError(f"{kind.value} Pod requires content, text or capabilities")
        if kind is PodType.TRANSPORT:
            required = {"capabilities", "link_contract"}
            if not required <= set(payload):
                raise ValueError("transport Pod requires capabilities and link_contract")
        if kind is PodType.RUNTIME:
            if not payload.get("interface") or not payload.get("capabilities"):
                raise ValueError("runtime Pod requires interface and capabilities")
        if kind is PodType.EMBEDDING and not payload.get("embedding_model"):
            raise ValueError("embedding Pod requires embedding_model")
        if kind is PodType.QUANTIZATION and not payload.get("quantization_method"):
            raise ValueError("quantization Pod requires quantization_method")


def pod_type_for_task(task: str) -> PodType:
    if task.startswith("deadline") or "buffer" in task:
        return PodType.MATH
    if task in {"lookup", "followup_lookup", "stale_followup", "missing"}:
        return PodType.CONTEXT
    if task in {"tool_call", "route", "expert_route"}:
        return PodType.ROUTER
    if task in {"pod_link", "transport", "attestation"}:
        return PodType.TRANSPORT
    if task in {"duplex", "realtime", "speculative_decode"}:
        return PodType.RUNTIME
    if task in {"ocr", "vision"}:
        return PodType.VISION
    if task in {"embedding", "rerank"}:
        return PodType.EMBEDDING
    if task in {"quantization", "loftq", "qat"}:
        return PodType.QUANTIZATION
    if task in {"replay_policy", "rsi", "policy"}:
        return PodType.POLICY
    return PodType.CONTEXT


def typed_artifact(registry, kind: str, pod_type: PodType | str, payload: Mapping[str, Any],
                   parents, principal: str = "local") -> str:
    """Create a Registry artifact with a validated, explicit Pod contract."""
    semantic_type = PodType(pod_type)
    body = dict(payload)
    body["semantic_type"] = semantic_type.value
    validate_payload(semantic_type, body)
    return registry.artifact(kind, body, parents, principal=principal)
