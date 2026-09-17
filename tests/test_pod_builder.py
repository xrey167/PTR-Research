from neural_pods.pod_builder import PodBuilder, PodMetadata, resolve_pod_identity
from neural_pods.registry import Registry
from neural_pods.pod_types import PodType


def test_builder_composes_granular_metadata_and_stable_branch_identity(tmp_path):
    reg = Registry(tmp_path / "r.sqlite3")
    origin = reg.origin("sap", "po-17", "v1", {"lead_time": 24})
    gen1 = reg.publish("supplier:muller:x12:lead_time", {"value": 24, "unit": "days"}, [origin])
    builder = PodBuilder(reg, pod_type=PodType.CONTEXT, name="muller-x12",
        metadata=PodMetadata(semantic_role="supplier_metric", domain="procurement",
            tags=("lead-time", "supplier"), aliases=("Müller delivery",),
            capabilities=("lookup", "temporal-update"), entity_ids=("supplier:muller", "component:x12"),
            sensitivity="internal", confidence=.98), contract={"content": "Supplier Müller needs 24 days for X12"})
    first = builder.build(parents=[gen1], kind="text")
    gen2 = reg.publish("supplier:muller:x12:lead_time", {"value": 18, "unit": "days"}, [origin])
    second = builder.branch(first, parents=[gen2])
    assert first != second
    assert reg.node(first)["payload"]["payload"]["pod_identity"] == reg.node(second)["payload"]["payload"]["pod_identity"]
    assert resolve_pod_identity(reg, builder.pod_identity) == second
    reg.close()


def test_metadata_rejects_invalid_confidence():
    try:
        PodMetadata(semantic_role="x", domain="y", confidence=1.1)
    except ValueError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("invalid confidence accepted")
