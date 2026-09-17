from research.evaluate_learned_dragonfly import run


def test_learned_dragonfly_generalizes_heldout_formulations():
    result = run()
    assert result["heldout_questions"] == 12
    assert result["top1_accuracy"] == 1.0
    assert result["negative_mean_score"] < 0.1
