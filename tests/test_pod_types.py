import pytest

from neural_pods.pod_types import ModelVariant, PodHeader, PodType, pod_type_for_task, typed_artifact, validate_payload
from neural_pods.registry import Registry
from neural_pods.execution_manifest import ExecutionManifest
from neural_pods.model_pod import ModelPodRuntime


def test_type_specific_contracts():
    validate_payload(PodType.CONTEXT, {"content": "supplier evidence", "surprise_gate": {"novelty_threshold": .7, "contradiction_threshold": .8}})
    validate_payload(PodType.MATH, {"expression": "lead + buffer", "operands": {"lead": 24, "buffer": 4}, "unit": "days"})
    validate_payload(PodType.MODEL, {"reader_identity": "r", "adapter_sha256": "a", "input_schema": "x", "output_schema": "y"})
    assert pod_type_for_task("deadline_weeks_short") is PodType.MATH
    assert pod_type_for_task("followup_lookup") is PodType.CONTEXT


def test_model_variants_distilled_and_moe_have_explicit_contracts():
    common = {"reader_identity": "r", "input_schema": "x", "output_schema": "y"}
    validate_payload(PodType.MODEL, {**common, "model_variant": ModelVariant.DISTILLED,
                                     "model_sha256": "student", "teacher_identity": "teacher-v1"})
    validate_payload(PodType.MODEL, {**common, "model_variant": ModelVariant.MOE,
                                     "model_sha256": "router", "experts": ["e0", "e1"]})
    with pytest.raises(ValueError):
        validate_payload(PodType.MODEL, {**common, "model_variant": ModelVariant.MOE,
                                         "model_sha256": "router"})
    with pytest.raises(ValueError):
        validate_payload(PodType.MODEL, {**common, "model_variant": "diffusion",
                                         "model_sha256": "x"})


def test_contract_rejects_untyped_or_incomplete_payloads():
    with pytest.raises(ValueError):
        validate_payload("math", {"expression": "x", "operands": []})
    with pytest.raises(ValueError):
        validate_payload("model", {"adapter_sha256": "a"})


def test_header_preserves_multi_source_generation_dag():
    node = {"id": "artifact:pod", "payload": {
        "origin_keys": ["o1", "o2"],
        "generations": {"g1": {"knowledge_key": "k1"}, "g2": {"knowledge_key": "k2"}},
        "payload": {"semantic_type": "model", "model_variant": "moe", "reader_identity": "r", "model_sha256": "a", "experts": ["e0"], "input_schema": "i", "output_schema": "o"}}}
    header = PodHeader.from_artifact("artifact:pod", node)
    assert header.knowledge_keys == ("k1", "k2")
    assert header.generation_keys == ("g1", "g2")
    assert header.knowledge_key is None and header.generation_key is None
    assert header.model_variant == "moe"


def test_typed_artifact_round_trips_registry_envelope(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    origin_a = registry.origin("sap", "17", "1", {"lead": 24})
    origin_b = registry.origin("analyst", "104", "1", {"risk": "high"})
    knowledge_a = registry.publish("supplier:muller:lead", {"subject": "muller", "predicate": "lead_time"}, [origin_a])
    knowledge_b = registry.publish("supplier:muller:risk", {"subject": "muller", "predicate": "risk"}, [origin_b])
    artifact = typed_artifact(registry, "lora", "model",
        {"reader_identity": "reader", "adapter_sha256": "adapter", "input_schema": "pod:v1", "output_schema": "text:v1"},
        [knowledge_a, knowledge_b])
    header = PodHeader.from_artifact(artifact, registry.node(artifact))
    assert header.generation_keys and len(header.generation_keys) == 2
    assert set(header.origin_keys) == {origin_a, origin_b}
    registry.close()


@pytest.mark.parametrize("variant,extra,operation", [
    (ModelVariant.LORA, {"adapter_sha256": "adapter"}, "attach_lora"),
    (ModelVariant.DISTILLED, {"model_sha256": "student", "teacher_identity": "teacher"}, "load_student"),
    (ModelVariant.MOE, {"model_sha256": "router", "experts": ["e0", "e1"]}, "load_moe"),
    (ModelVariant.QUANTIZED, {"model_sha256": "q4"}, "load_quantized"),
    (ModelVariant.BASE, {"model_sha256": "base"}, "load_base"),
])
def test_model_pod_runtime_emits_variant_specific_plan(tmp_path, variant, extra, operation):
    registry = Registry(tmp_path / f"{variant}.sqlite")
    origin = registry.origin("model", str(variant), "1", {"variant": variant.value})
    generation = registry.publish("model:reader", {"subject": "reader", "predicate": "variant"}, [origin])
    payload = {"reader_identity": "reader", "model_variant": variant.value,
               "model_family": "qwen3.5",
               "post_training": "routing-guided-agentic",
               "interface": "text->text",
               "router_ref": "llm-semantic-router/Vela-1.0-Encoder-307M",
               "ranker_ref": "llm-semantic-router/mmbert-rerank-32k-2d-matryoshka",
               "stage_role": "reader",
               "parallel_strategy": {"tensor_parallel": 1},
               "cache_policy": {"mode": "pod_scoped", "max_tokens": 4096},
               "input_schema": "pod:v1", "output_schema": "text:v1", **extra}
    reader = typed_artifact(registry, "model", PodType.MODEL, payload, [generation])
    manifest = ExecutionManifest.build(registry, generation, [reader], reader_key=reader)
    plan = ModelPodRuntime(registry).prepare(manifest, base_model="Qwen/Qwen2.5-3B")
    assert plan["operation"] == operation
    assert plan["model_family"] == "qwen3.5"
    assert plan["post_training"] == "routing-guided-agentic"
    assert plan["interface"] == "text->text"
    assert plan.get("router") == "llm-semantic-router/Vela-1.0-Encoder-307M" or plan.get("pod_router") == "llm-semantic-router/Vela-1.0-Encoder-307M"
    assert plan["ranker"] == "llm-semantic-router/mmbert-rerank-32k-2d-matryoshka"
    assert plan["stage_role"] == "reader"
    assert plan["parallel_strategy"] == {"tensor_parallel": 1}
    assert plan["generation_key"] == generation
    registry.close()
