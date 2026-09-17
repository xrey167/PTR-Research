from research.evaluate_hard_benchmark import run


def test_hard_benchmark_exposes_metadata_routing_gain():
    report = run()
    assert report["baselines"]["tfidf"]["recall_at_3"] == 0.86
    assert report["baselines"]["provenance_router"]["recall_at_3"] == 1.0
    assert report["scope"].startswith("leakage-controlled")
