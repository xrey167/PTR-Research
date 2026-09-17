"""Register a completed reader LoRA as a typed, lineage-bearing model Pod."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.pod_types import typed_artifact
from neural_pods.registry import Registry


def file_sha(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def register(registry_path: Path, adapter_run: Path, training_inputs: Path, principal: str = "buyer") -> dict:
    report = json.loads((adapter_run / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "trained_not_evaluated" or not report.get("reader_identity"):
        raise ValueError("adapter run is not a completed reader training artifact")
    protocol = json.loads((training_inputs / "protocol.json").read_text(encoding="utf-8"))
    input_hash = file_sha(training_inputs / "protocol.json")
    registry = Registry(registry_path)
    try:
        origin = registry.origin("reader-training", training_inputs.name, protocol["model_revision"],
                                {"protocol_sha256": input_hash, "rows": protocol["rows"],
                                 "adapter_run": adapter_run.name}, acl=[principal])
        knowledge_key = "model:reader:qwen2.5-3b"
        generation = registry.publish(knowledge_key, {
            "subject": "reader", "predicate": "typed-pod-generation", "type": "MODEL",
            "model_repo": protocol["model_repo"], "model_revision": protocol["model_revision"]},
            [origin], principal=principal, acl=[principal])
        payload = {"schema": "research-reader:v1", "semantic_type": "model",
                   "reader_identity": report["reader_identity"],
                   "adapter_sha256": report["adapter_files"].get("adapter_model.safetensors"),
                   "adapter_run": adapter_run.name, "protocol_sha256": input_hash,
                   "input_schema": "typed-pod:v1", "output_schema": "reader-answer:v1",
                   "adapter_files": report["adapter_files"],
                   "training_rows": protocol["rows"], "optimizer_updates": protocol["optimizer_updates"]}
        artifact = typed_artifact(registry, "lora", "model", payload, [generation], principal=principal)
        result = {"origin_key": origin, "knowledge_key": knowledge_key,
                  "generation_key": generation, "artifact_key": artifact,
                  "reader_identity": report["reader_identity"], "protocol_sha256": input_hash,
                  "full_research_goal_complete": False}
        (adapter_run / "registry-registration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        registry.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path, required=True)
    parser.add_argument("--training-inputs", type=Path, required=True)
    parser.add_argument("--principal", default="buyer")
    args = parser.parse_args()
    print(json.dumps(register(args.registry, args.adapter_run, args.training_inputs, args.principal), indent=2))


if __name__ == "__main__":
    main()
