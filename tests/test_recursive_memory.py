from neural_pods.recursive_memory import BroadThenDeepExplorer, DiscoveryReplay, Experience
from neural_pods.pod_types import typed_artifact
from neural_pods.registry import Registry, InvalidState


def test_replay_scores_only_verified_history():
    rows = [
        Experience("r1", "browser", "login", "submit", "ok", True, metadata={"confidence": "0.9"}),
        Experience("r2", "browser", "login", "submit", "ok", True, metadata={"confidence": "0.8"}),
        Experience("bad", "browser", "login", "submit", "error", False),
    ]
    score = DiscoveryReplay(rows).score(lambda condition, action: "ok")
    assert (score.evaluated, score.correct, score.accuracy) == (2, 2, 1.0)


def test_broad_then_deep_expands_weak_branch_and_consolidates_verified_pod():
    seed = Experience("root", "shell", "install", "run", "unknown", False, metadata={"confidence": "0.2"})
    child = Experience("child", "shell", "install", "run", "success", True, depth=1,
                       parent_id="root", metadata={"confidence": "0.95"})
    explorer = BroadThenDeepExplorer(max_depth=2, confidence_threshold=0.75)
    rows = explorer.explore([seed], lambda parent: [child] if parent.experience_id == "root" else [])
    pods = explorer.consolidate(rows)
    assert [x.experience_id for x in rows] == ["root", "child"]
    assert len(pods) == 1
    assert pods[0].to_payload()["experience"]["source_experience_ids"] == ["child"]


def test_discovery_tree_replays_parent_child_history():
    rows = [
        Experience("root", "os", "open", "click", "opened", True),
        Experience("child", "os", "opened", "save", "saved", True, parent_id="root", depth=1),
    ]
    assert [x.experience_id for x in DiscoveryReplay(rows).discovery_tree("root")] == ["root", "child"]


def test_consolidated_experience_pod_obeys_registry_revocation(tmp_path):
    rows = [Experience("e1", "os", "open", "click", "ok", True)]
    pod = BroadThenDeepExplorer().consolidate(rows)[0]
    registry = Registry(tmp_path / "experience.sqlite")
    origin = registry.origin("experience", "os", "1", {"id": "e1"})
    generation = registry.publish("experience:os", {"subject": "os", "predicate": "experience"}, [origin])
    artifact = typed_artifact(registry, "text", "context", pod.to_payload(), [generation])
    assert registry.snapshot([artifact]).artifacts == (artifact,)
    registry.revoke(origin)
    try:
        registry.snapshot([artifact])
    except InvalidState:
        pass
    else:
        raise AssertionError("revoked Experience-Pod remained activatable")
    registry.close()
