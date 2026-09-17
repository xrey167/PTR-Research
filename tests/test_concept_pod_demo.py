from research.demo_concept_pod import run


def test_concept_pod_demo_activation_and_revocation():
    result = run()
    assert result["activation_changed_model"]
    assert result["revocation_restored_baseline"]
    assert result["pod"]["generation_key"] == "generation:8"
