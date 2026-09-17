"""Create a generation-bound Model-Pod for the newly compiled topic mix."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.registry import Registry
from neural_pods.pod_types import typed_artifact


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "topic-model-pod-generation3-full-001"
    out.mkdir(parents=True, exist_ok=True)
    registry = Registry(out / "registry.sqlite3")
    source = registry.origin("neural-pods", "typed-topic-mix", "1",
        {"topics": ["pod taxonomy", "namespace cache", "vLLM Omni stages"],
         "dataset": "runs/pod-training-mix-001/dataset.json"})
    generation = registry.publish("main-model:typed-topics",
        {"subject": "typed pod operations", "predicate": "supports",
         "objects": ["taxonomy", "namespace-cache", "pipeline-stages"]}, [source])
    adapter = root / "runs" / "topic-lora-gpu-generation3-full-001" / "adapter"
    adapter_hash = __import__("hashlib").sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    pod = typed_artifact(registry, "lora", "model", {
        "reader_identity": "ed4bfc55f928db06ee291a079d1ba8963d1e750b6bb845123c6b2205bbe35731",
        "adapter_sha256": adapter_hash, "model_variant": "lora",
        "input_schema": "typed-pod:v2", "output_schema": "answer:v2",
        "model_family": "Qwen2.5-3B", "post_training": "typed-topic-mix-prepared",
        "interface": "text->text", "router_ref": "llm-semantic-router/Vela-1.0-Encoder-307M",
        "ranker_ref": "llm-semantic-router/mmbert-rerank-32k-2d-matryoshka",
        "stage_role": "reader", "pod_identity": "pod:main-model-typed-topics",
        "pod_name": "main-model-typed-topics", "semantic_role": "knowledge-reader",
        "domain": "neural-pods", "tags": ["pod:model", "role:execution", "has:interface", "typed-topics"],
    }, [generation])
    report = {"status": "linked_existing_adapter_topic_generation3_full", "source": source,
              "generation": generation, "pod_artifact": pod,
              "pod_identity": "pod:main-model-typed-topics", "adapter_path": str(adapter),
              "adapter_sha256": adapter_hash, "training_report": "runs/topic-lora-gpu-generation3-full-001/report.json",
              "training_mix": "runs/pod-training-mix-001/report.json",
              "training_status": "trained_on_xrserver_gpu"}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    registry.close()


if __name__ == "__main__":
    main()
