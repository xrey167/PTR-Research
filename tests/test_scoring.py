from run_experiment import correct
from verify_saved import numeric_correct


def test_exact_match_does_not_accept_wrong_or_ambiguous_numbers():
    assert correct("24 days.", "24 days")
    assert not correct("124 days", "24 days")
    assert not correct("24 or 18 days", "24 days")
    assert not correct("24", "24 days")


def test_numeric_accuracy_is_separate_from_output_format():
    assert numeric_correct("24", 24)
    assert numeric_correct("24 Tage", 24)
    assert not numeric_correct("124", 24)
    assert not numeric_correct("24 or 18 days", 24)
