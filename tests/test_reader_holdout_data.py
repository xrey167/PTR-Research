"""The holdout split must stay uncontaminated and regenerable.

Both properties are what the frozen splits lost: the frozen dev/test bundles
hold 132 cases while today's generator produces 96, so they cannot be rebuilt
from the repository at all.
"""
import json
from collections import Counter

import generate_holdout_split as generator
from research.reader_holdout_data import (
    build_holdout, disjointness_report)
from research.reader_training_data import build_data


def test_no_entity_value_or_phrasing_is_shared_with_the_frozen_splits():
    report = disjointness_report()
    assert report["overlaps"] == {"ids": [], "questions": [], "suppliers": [],
                                  "components": [], "values": []}
    assert report["disjoint"] is True


def test_case_families_match_the_frozen_evaluation_families():
    """Same families means the score stays comparable to the frozen runs.
    pod_kind is train-only in the frozen data, so an eval split has none."""
    holdout = {row["task"] for row in build_holdout()}
    frozen_test = {row["task"] for row in build_data()["test"]}
    assert holdout == frozen_test


def test_split_is_balanced_across_languages_and_families():
    rows = build_holdout()
    assert Counter(r["language"] for r in rows) == {"en": 48, "de": 48}
    counts = Counter(r["task"] for r in rows)
    assert counts["lookup"] == counts["buffer"] == 8
    assert counts["missing"] == 4          # one per entity, not one per value


def test_only_missing_cases_lack_evidence_and_answer_unknown():
    for row in build_holdout():
        if row["task"] == "missing":
            assert row["evidence"] is None and row["target"] == "UNKNOWN"
        else:
            assert row["evidence"] is not None and row["target"] != "UNKNOWN"


def test_ids_are_unique():
    ids = [row["id"] for row in build_holdout()]
    assert len(ids) == len(set(ids))


def test_checked_in_artifact_matches_the_generator():
    payload, manifest = generator.render()
    assert generator.SPLIT_FILE.read_text(encoding="utf-8") == payload
    stored = json.loads(generator.MANIFEST_FILE.read_text(encoding="utf-8"))
    assert stored["cases_sha256"] == manifest["cases_sha256"]
    assert stored["generator_sha256"] == manifest["generator_sha256"]
    assert stored["rows"] == manifest["rows"] == 96
