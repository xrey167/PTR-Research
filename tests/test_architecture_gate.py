"""Tests for the gate's evidence handling.

The gate compares recorded numbers; these tests cover how it behaves when a
recorded file is absent, which used to be indistinguishable from a threshold
violation and in one case let a check pass with no evidence at all.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import verify_architecture_gate as gate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "research" / "runs"
ARCHITECTURE = "architecture-20260917.json"


@pytest.fixture()
def evidence_dir(tmp_path):
    """A writable copy of the checked-in evidence."""
    target = tmp_path / "runs"
    shutil.copytree(EVIDENCE, target)
    return target


def _failures(evidence_dir):
    with pytest.raises(SystemExit) as caught:
        gate.verify(evidence_dir / ARCHITECTURE)
    message = str(caught.value)
    failed = message.splitlines()[0].split(": ", 1)[1].split(", ")
    return failed, message


def test_evidence_is_resolved_next_to_the_run_file(evidence_dir):
    found = gate._find("mesh-presence-20260920.json", gate._search_dirs(
        evidence_dir / ARCHITECTURE))
    assert found is not None and found.parent == evidence_dir


def test_missing_file_is_recorded_not_raised(tmp_path):
    missing = []
    assert gate._read("gibt-es-nicht.json", [tmp_path], missing) == {}
    assert gate._read_lines("auch-nicht.jsonl", [tmp_path], missing) == []
    assert missing == ["gibt-es-nicht.json", "auch-nicht.jsonl"]


def test_absent_evidence_is_reported_apart_from_a_broken_threshold(evidence_dir):
    (evidence_dir / "mesh-presence-20260920.json").unlink()
    failed, message = _failures(evidence_dir)
    assert "mesh_presence" in failed
    assert "missing evidence" in message
    assert "mesh-presence-20260920.json" in message
    assert "not a regression" in message


def test_comparative_check_fails_when_only_its_baseline_is_missing(evidence_dir):
    """gen6_promoted_dev compares gen6 against gen4. With gen4 absent the
    comparison is `>= 0` and used to pass on no evidence."""
    baseline = "qwen3b-eval-dev-gen4-adapter-20260917-report.json"
    (evidence_dir / baseline).write_text(
        '{"status": "completed", "reader_unchanged": true, '
        '"exact_target_matches": 119, "guarded_exact_target_matches": 92}',
        encoding="utf-8")
    failed, _ = _failures(evidence_dir)
    assert "gen6_promoted_dev" not in failed     # both sides present: real compare

    (evidence_dir / baseline).unlink()
    failed, message = _failures(evidence_dir)
    assert "gen6_promoted_dev" in failed
    assert baseline in message


def test_threshold_violation_is_not_labelled_missing(evidence_dir):
    (evidence_dir / "mesh-presence-20260920.json").write_text(
        '{"discovery": {"discovered": true}, "rounds_ok": 99, '
        '"rtt_p50_ms": 0.3}', encoding="utf-8")
    failed, message = _failures(evidence_dir)
    assert "mesh_presence" in failed
    assert "mesh-presence-20260920.json" not in message.split("missing evidence")[-1]
