from research.vela_domain_gate import classify


def test_vela_gate_is_advisory_and_thresholded():
    gate = lambda _q, **_kw: [{"label": "business", "score": 0.91}, {"label": "law", "score": 0.09}]
    result = classify(gate, "approval policy")
    assert result == {"label": "business", "score": 0.91, "advisory": True, "confident": True}
