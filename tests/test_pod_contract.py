import pytest

from neural_pods.pod_contract import LifecycleGate, PodLink, PodManifest
from neural_pods.registry import InvalidState


def manifest(status="active"):
    return PodManifest("pod:reader", "g8", "artifact:reader:g8", "reader", "Qwen3-4B-SFT",
                       "Qwen3-4B", "openai_chat", ("grounded_answer",), ("domain:research",),
                       "ns:research", ("buyer",), status, {"origin_key": "src:research:1"})


def test_link_attestation_and_generation_are_checked():
    target = manifest()
    link = PodLink("trace:1", "pod:research", target.pod_id, target.generation,
                   target.artifact_id, "ssh_tunnel", "grounded_answer", ("buyer",),
                   attestation=target.manifest_hash, visited=("pod:research",))
    assert link.validate(target, principal="buyer")
    with pytest.raises(InvalidState):
        PodLink("trace:2", "pod:research", target.pod_id, "g7", target.artifact_id,
                "ssh_tunnel", "grounded_answer", ("buyer",)).validate(target, principal="buyer")


def test_link_rejects_cycles_and_expired_budget():
    target = manifest()
    with pytest.raises(InvalidState):
        PodLink("trace:1", "pod:research", target.pod_id, target.generation,
                target.artifact_id, "ssh_tunnel", "grounded_answer", ("buyer",),
                visited=(target.pod_id,)).validate(target, principal="buyer")
    with pytest.raises(InvalidState):
        PodLink("trace:1", "pod:research", target.pod_id, target.generation,
                target.artifact_id, "ssh_tunnel", "grounded_answer", ("buyer",),
                hop_budget=1, visited=("pod:a",)).validate(target, principal="buyer")


def test_lifecycle_gate_requires_checks():
    candidate = manifest("candidate")
    trained = LifecycleGate.transition(candidate, "trained", checks={})
    evaluated = LifecycleGate.transition(trained, "evaluated", checks={
        "manifest_hash": True, "provenance": True, "base_identity": True})
    approved = LifecycleGate.transition(evaluated, "approved", checks={
        "manifest_hash": True, "provenance": True, "base_identity": True})
    assert LifecycleGate.transition(approved, "active", checks={
        "manifest_hash": True, "provenance": True, "base_identity": True}).status == "active"
