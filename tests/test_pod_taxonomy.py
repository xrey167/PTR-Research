import pytest
from neural_pods.pod_taxonomy import PodDescriptor, route_candidates, REQUIRED_TRAINING_TAGS, validate_training_record


def test_taxonomy_filters_types_and_tags_before_soft_domain():
    pods = [
        PodDescriptor("ctx", "context", "business", frozenset({"supplier", "alias"})),
        PodDescriptor("math", "math", "business", frozenset({"lead_time", "numeric"})),
        PodDescriptor("reason", "reasoning", "business", frozenset({"multi_hop"})),
        PodDescriptor("law", "model", "law", frozenset({"policy"})),
    ]
    result = route_candidates(pods, domain="business", required_tags=("lead_time",), soft_domain=("law",))
    assert [p.artifact_key for p in result] == ["math"]


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        PodDescriptor("bad", "answer", "business")


def test_training_contract_requires_type_specific_tags_for_every_pod_kind():
    for kind, required in REQUIRED_TRAINING_TAGS.items():
        row = {"id": kind, "pod_type": kind, "domain": "test", "semantic_role": "role",
               "tags": list(required), "origin_keys": ["o"], "knowledge_key": "k",
               "generation_key": "g", "split": "train", "input": "q", "target": "a"}
        assert validate_training_record(row) is row
        row["tags"] = []
        with pytest.raises(ValueError, match="required tags"):
            validate_training_record(row)
