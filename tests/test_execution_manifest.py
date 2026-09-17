import pytest

from neural_pods.execution_manifest import ExecutionManifest
from neural_pods.registry import InvalidState, Registry


def setup(tmp_path):
    r = Registry(tmp_path / "r.db")
    origin = r.origin("sap", "po-1", "1", {"value": 24})
    generation = r.publish("supplier:muller:x12:lead_time", {"value": 24}, [origin], principal="buyer")
    vector = r.artifact("vector", {"embedding": [1.0]}, [generation], principal="buyer")
    return r, origin, generation, vector


def test_manifest_validates_lineage_and_serializes(tmp_path):
    r, _, generation, vector = setup(tmp_path)
    manifest = ExecutionManifest.build(r, generation, [vector], principal="buyer")
    assert manifest.validate(r).artifacts == manifest.snapshot.artifacts
    assert manifest.as_dict()["generation_key"] == generation
    r.close()


def test_manifest_rejects_stale_generation_and_revocation(tmp_path):
    r, origin, generation, vector = setup(tmp_path)
    manifest = ExecutionManifest.build(r, generation, [vector], principal="buyer")
    replacement = r.publish("supplier:muller:x12:lead_time", {"value": 18}, [origin], principal="buyer")
    with pytest.raises(InvalidState):
        manifest.validate(r)
    fresh = r.artifact("vector", {"embedding": [0.8]}, [replacement], principal="buyer")
    current = ExecutionManifest.build(r, replacement, [fresh], principal="buyer")
    r.revoke(origin)
    with pytest.raises(InvalidState):
        current.validate(r)
    r.close()


def test_manifest_reader_must_share_generation_lineage(tmp_path):
    r, origin, generation, vector = setup(tmp_path)
    training = r.origin("training", "reader", "1", {"run": "fixture"}, acl=("buyer",))
    reader = r.artifact("lora", {"schema": "value-bearing-qwen-lora:v1"},
                        [generation, training], principal="buyer")
    manifest = ExecutionManifest.build(r, generation, [vector], reader_key=reader, principal="buyer")
    assert manifest.reader_key == reader
    other_generation = r.publish("other:key", {"value": 1}, [origin], principal="buyer")
    with pytest.raises(InvalidState, match="Reader binding is not derived"):
        ExecutionManifest.build(r, other_generation, [r.artifact("vector", {"embedding": [0.2]}, [other_generation], principal="buyer")], reader_key=reader, principal="buyer")
    r.close()


def test_manifest_accepts_generation_bound_model_variant(tmp_path):
    r, _, generation, vector = setup(tmp_path)
    model = r.artifact("model", {"semantic_type": "model", "model_variant": "moe",
                                  "model_sha256": "router", "experts": ["e0"],
                                  "reader_identity": "reader", "input_schema": "i",
                                  "output_schema": "o"}, [generation], principal="buyer")
    manifest = ExecutionManifest.build(r, generation, [vector], reader_key=model, principal="buyer")
    assert manifest.reader_key == model
    r.close()
