from pathlib import Path
import pytest
from neural_pods.registry import Registry, InvalidState, hash_files


def test_empty_or_missing_adapter_is_never_valid(tmp_path):
    with pytest.raises(InvalidState): hash_files(tmp_path)
    with pytest.raises(InvalidState): hash_files(tmp_path / "missing")


def test_revocation_persists_after_restart(tmp_path):
    path = tmp_path / "state.db"
    r = Registry(path)
    origin = r.origin("sap", "1", "v1", "evidence")
    knowledge = r.publish("k", {"value": 24}, [origin])
    pod = r.artifact("lora", {"weights": "test"}, [knowledge])
    snapshot = r.snapshot([pod])
    r.revoke(origin)
    r.close()
    reopened = Registry(path)
    with pytest.raises(InvalidState): reopened.snapshot([pod])
    with pytest.raises(InvalidState): reopened.commit(snapshot, "24 days")
    reopened.close()


def test_derived_knowledge_cannot_outlive_parent_generation(tmp_path):
    r = Registry(tmp_path / "state.db")
    origin = r.origin("sap", "1", "v1", "evidence")
    parent = r.publish("supplier:lead_time", {"value": 24}, [origin])
    derived = r.publish("supplier:risk", {"value": "high"}, [parent])
    pod = r.artifact("lora", {}, [derived])
    r.snapshot([pod])
    source2 = r.origin("sap", "1", "v2", "new evidence")
    r.publish("supplier:lead_time", {"value": 18}, [source2])
    with pytest.raises(InvalidState): r.snapshot([pod])
    r.close()
