"""The balanced routing dataset must be present and match its generator.

The artifact used to live only under runs/, which .gitignore excludes, so
this test could never pass from a clone. It is pure code output, so it is
now generated into research/runs/ and shipped; runs/ stays as a fallback for
the server's own copy.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from generate_taxonomy_dataset import build, render

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [REPO / "research" / "runs" / "taxonomy-routing-balanced-003.jsonl",
              REPO / "runs" / "taxonomy-routing-balanced-003.jsonl"]
POD_TYPES = ("context", "math", "model", "reasoning", "retrieval")


@pytest.fixture(scope="module")
def dataset_path():
    for candidate in CANDIDATES:
        if candidate.exists():
            return candidate
    raise AssertionError(
        "no taxonomy dataset found; regenerate it with "
        "`python research/generate_taxonomy_dataset.py`")


def test_balanced_taxonomy_dataset(dataset_path):
    rows = [json.loads(line) for line in
            dataset_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 500
    assert Counter(r["pod_type"] for r in rows) == {t: 100 for t in POD_TYPES}
    for pod_type in POD_TYPES:
        assert Counter(r["split"] for r in rows if r["pod_type"] == pod_type) == {
            "train": 60, "validation": 20, "test": 20}


def test_checked_in_dataset_matches_the_generator(dataset_path):
    assert dataset_path.read_text(encoding="utf-8") == render()


def test_heldout_phrasings_never_appear_in_training_rows():
    rows = build()
    train = {r["question"] for r in rows if r["split"] == "train"}
    heldout = {r["question"] for r in rows if r["split"] != "train"}
    assert not (train & heldout)
