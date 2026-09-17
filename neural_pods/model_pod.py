"""Runtime-neutral activation plans for typed Model-Pods.

The loader deliberately returns a plan instead of importing a particular model
server.  vLLM, Transformers/PEFT, or another runtime can execute the same
lineage-checked plan without changing the registry contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .execution_manifest import ExecutionManifest
from .pod_types import ModelVariant, PodType, validate_payload
from .registry import InvalidState
from .pod_builder import resolve_pod_identity


@dataclass(frozen=True)
class ModelPodSpec:
    artifact_key: str
    variant: ModelVariant
    reader_identity: str
    input_schema: str
    output_schema: str
    model_ref: str | None = None
    adapter_ref: str | None = None
    teacher_identity: str | None = None
    experts: tuple[str, ...] = ()
    model_family: str | None = None
    post_training: str | None = None
    interface: str | None = None
    router_ref: str | None = None
    ranker_ref: str | None = None
    stage_role: str | None = None
    parallel_strategy: Mapping[str, Any] | None = None
    cache_policy: Mapping[str, Any] | None = None

    @classmethod
    def from_registry(cls, registry, artifact_key: str) -> "ModelPodSpec":
        node = registry.node(artifact_key)
        if node["kind"] not in {"model", "lora"}:
            raise InvalidState("Model-Pod runtime requires a model or lora artifact")
        envelope = node["payload"]
        payload = envelope.get("payload", envelope)
        if payload.get("semantic_type") != PodType.MODEL.value:
            raise InvalidState("Artifact is not typed as a model Pod")
        validate_payload(PodType.MODEL, payload)
        try:
            variant = ModelVariant(payload.get("model_variant", ModelVariant.LORA.value))
        except ValueError as exc:
            raise InvalidState("Unknown Model-Pod variant") from exc
        return cls(
            artifact_key=artifact_key,
            variant=variant,
            reader_identity=payload["reader_identity"],
            input_schema=payload["input_schema"],
            output_schema=payload["output_schema"],
            model_ref=payload.get("model_ref") or payload.get("model_sha256"),
            adapter_ref=payload.get("adapter_ref") or payload.get("adapter_sha256"),
            teacher_identity=payload.get("teacher_identity"),
            experts=tuple(payload.get("experts", ())),
            model_family=payload.get("model_family"),
            post_training=payload.get("post_training"),
            interface=payload.get("interface"),
            router_ref=payload.get("router_ref"), ranker_ref=payload.get("ranker_ref"),
            stage_role=payload.get("stage_role"), parallel_strategy=payload.get("parallel_strategy"),
            cache_policy=payload.get("cache_policy"),
        )

    def activation_plan(self, *, base_model: str | None = None) -> Mapping[str, Any]:
        """Return a deterministic server-neutral activation instruction."""
        if self.variant is ModelVariant.LORA:
            if not self.adapter_ref:
                raise InvalidState("LoRA Model-Pod has no adapter reference")
            return {"operation": "attach_lora", "base_model": base_model,
                    "adapter": self.adapter_ref, "reader_identity": self.reader_identity,
                    "model_family": self.model_family, "post_training": self.post_training,
                    "interface": self.interface, "router": self.router_ref,
                    "ranker": self.ranker_ref, "stage_role": self.stage_role,
                    "parallel_strategy": self.parallel_strategy, "cache_policy": self.cache_policy}
        if self.variant is ModelVariant.DISTILLED:
            return {"operation": "load_student", "model": self.model_ref,
                    "teacher_identity": self.teacher_identity,
                    "reader_identity": self.reader_identity, "model_family": self.model_family,
                    "post_training": self.post_training, "interface": self.interface,
                    "router": self.router_ref, "ranker": self.ranker_ref, "stage_role": self.stage_role,
                    "parallel_strategy": self.parallel_strategy, "cache_policy": self.cache_policy}
        if self.variant is ModelVariant.MOE:
            return {"operation": "load_moe", "router_model": self.model_ref,
                    "experts": list(self.experts), "reader_identity": self.reader_identity,
                    "model_family": self.model_family, "post_training": self.post_training,
                    "interface": self.interface, "pod_router": self.router_ref, "ranker": self.ranker_ref,
                    "stage_role": self.stage_role, "parallel_strategy": self.parallel_strategy, "cache_policy": self.cache_policy}
        if self.variant is ModelVariant.QUANTIZED:
            return {"operation": "load_quantized", "model": self.model_ref,
                    "reader_identity": self.reader_identity, "model_family": self.model_family,
                    "post_training": self.post_training, "interface": self.interface,
                    "router": self.router_ref, "ranker": self.ranker_ref, "stage_role": self.stage_role,
                    "parallel_strategy": self.parallel_strategy, "cache_policy": self.cache_policy}
        return {"operation": "load_base", "model": self.model_ref,
                "reader_identity": self.reader_identity, "model_family": self.model_family,
                "post_training": self.post_training, "interface": self.interface,
                "router": self.router_ref, "ranker": self.ranker_ref, "stage_role": self.stage_role,
                "parallel_strategy": self.parallel_strategy, "cache_policy": self.cache_policy}


class ModelPodRuntime:
    """Resolve and validate a Model-Pod at the execution boundary."""

    def __init__(self, registry):
        self.registry = registry

    def prepare(self, manifest: ExecutionManifest, *, base_model: str | None = None) -> Mapping[str, Any]:
        if not manifest.reader_key:
            raise InvalidState("Model-Pod runtime requires a reader binding")
        manifest.validate(self.registry)
        spec = ModelPodSpec.from_registry(self.registry, manifest.reader_key)
        if manifest.generation_key not in self.registry.node(manifest.reader_key)["payload"].get("generations", {}):
            raise InvalidState("Model-Pod is not bound to the manifest generation")
        plan = dict(spec.activation_plan(base_model=base_model))
        plan.update({"artifact_key": spec.artifact_key,
                     "generation_key": manifest.generation_key,
                     "origin_keys": list(manifest.origin_keys),
                     "input_schema": spec.input_schema,
                     "output_schema": spec.output_schema,
                     "post_training": spec.post_training,
                     "interface": spec.interface})
        return plan

    def prepare_for_pod(self, manifest: ExecutionManifest, pod_identity: str,
                        *, base_model: str | None = None) -> Mapping[str, Any]:
        """Prepare using a stable Pod identity and resolve its current branch.

        Callers keep the model link as ``pod_identity``; only the lifecycle
        boundary selects the currently active immutable artifact.
        """
        artifact_key = resolve_pod_identity(self.registry, pod_identity,
                                             principal=manifest.snapshot.principal)
        rebound = ExecutionManifest.build(self.registry, manifest.generation_key,
                                          manifest.artifact_keys, reader_key=artifact_key,
                                          principal=manifest.snapshot.principal)
        return self.prepare(rebound, base_model=base_model)
