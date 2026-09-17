from research.generate_paper_benchmark import build


def test_paper_benchmark_has_frozen_splits_and_provenance():
    rows = build(700)
    assert len(rows) == 700
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert all(row["origin_keys"] and row["generation_key"] for row in rows)
    assert {row["pod_type"] for row in rows} == {"context", "math", "model"}
