import json

from research.run_paper_benchmarks import SPECS, build_manifest


def test_paper_manifest_has_pinned_cohorts_and_comparable_arms(tmp_path):
    out = tmp_path / "manifest.json"
    manifest = build_manifest("both", out)
    assert [(x.name, x.tasks) for x in SPECS] == [("osworld_v2", 108), ("agents_last_exam", 67)]
    assert manifest["arms"] == ["baseline", "rsi"]
    assert manifest["evaluation_protocol"]["verifier_outside_learning"]
    assert manifest["evaluation_protocol"]["memory_frozen_for_test"]
    assert manifest["status"] == "prepared_not_run"
    assert json.loads(out.read_text())["benchmarks"][0]["spec"]["release"] == "v2026.08.08"
