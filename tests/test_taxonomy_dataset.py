import json
from collections import Counter
from pathlib import Path


def test_balanced_taxonomy_dataset():
    path = Path("runs/taxonomy-routing-balanced-003.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 500
    assert Counter(r["pod_type"] for r in rows) == {t: 100 for t in ("context", "math", "model", "reasoning", "retrieval")}
    for pod_type in ("context", "math", "model", "reasoning", "retrieval"):
        assert Counter(r["split"] for r in rows if r["pod_type"] == pod_type) == {"train": 60, "validation": 20, "test": 20}


