from neural_pods.data import FACTS, questions


def test_question_families_are_disjoint():
    for fact in FACTS:
        train, validation, test = [set(questions(fact, s)) for s in ["train", "validation", "test"]]
        assert not (train & validation or train & test or validation & test)
        assert all(fact.answer not in q for q in train | validation | test)
