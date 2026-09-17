from research.reader_answer_guard import guarded_answer


def test_guard_recomputes_deadline_units_and_buffer():
    base = {"evidence": {"lead_time_days": 35}, "language": "en"}
    assert guarded_answer({**base, "question": "is a deadline of one week and 5 days sufficient?"}, "Yes; 35 days") == "No; 35 days"
    assert guarded_answer({**base, "question": "is a deadline of five weeks sufficient?"}, "No; 35 days") == "Yes; 35 days"
    assert guarded_answer({**base, "question": "what duration includes four extra buffer days?"}, "35 days") == "39 days"


def test_guard_handles_german_and_abstention():
    base = {"evidence": {"lead_time_days": 14}, "language": "de"}
    assert guarded_answer({**base, "question": "Genügt eine Frist von eine Woche und 5 Tage?"}, "Ja; 14 Tage") == "Nein; 14 Tage"
    assert guarded_answer({"evidence": None, "language": "de", "question": "Wie lange?"}, "14 Tage") == "UNKNOWN"
