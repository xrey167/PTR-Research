import json
from pathlib import Path

from neural_pods.pod_types import PodHeader
from neural_pods.registry import Registry
from research.register_reader_artifact import register


def test_reader_registration_creates_typed_lineage(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "protocol.json").write_text(json.dumps({"model_revision": "rev", "model_repo": "local",
        "rows": {"train": 1, "dev": 1, "test": 1}, "optimizer_updates": 1}), encoding="utf-8")
    run = tmp_path / "adapter"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"status": "trained_not_evaluated",
        "reader_identity": {"sha256": "reader"}, "adapter_files": {"adapter_model.safetensors": "adapter"}}), encoding="utf-8")
    result = register(tmp_path / "registry.sqlite", run, inputs)
    registry = Registry(tmp_path / "registry.sqlite")
    header = PodHeader.from_artifact(result["artifact_key"], registry.node(result["artifact_key"]))
    assert header.pod_type.value == "model"
    assert header.knowledge_keys == (result["knowledge_key"],)
    assert header.origin_keys == (result["origin_key"],)
    registry.close()
