"""Tests for the gate's evidence handling.

The gate compares recorded numbers; these tests cover how it behaves when a
recorded file is absent, which used to be indistinguishable from a threshold
violation and in one case let a check pass with no evidence at all.
"""
import shutil
from pathlib import Path

import pytest

import verify_architecture_gate as gate

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


# --- every evidence file must be load-bearing for the checks that read it ---
#
# `gen6_promoted_dev` compared gen6 against gen4 and, with the gen4 report
# gone, compared against a default of 0 — green on no evidence. A hand-kept
# table of such comparisons is the same kind of thing that failed in the
# first place, so the table is checked here rather than trusted: for every
# evidence file, the set of checks that go red when it disappears must be
# exactly the set of checks whose predicate reads it.


def _evidence_dependencies():
    """{check name -> {evidence filenames its predicate reads}}, from source."""
    import ast

    source = Path(gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "verify")

    # variable -> evidence file, from `x = _read("file", ...)` / `_read_lines`
    files: dict[str, str] = {}
    for node in ast.walk(function):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        called = getattr(node.value.func, "id", "")
        if called in ("_read", "_read_lines") and node.value.args:
            first = node.value.args[0]
            if isinstance(first, ast.Constant) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    files[target.id] = first.value

    # variables derived from an evidence variable inherit its file
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            for inner in ast.walk(node.value):
                if isinstance(inner, ast.Name) and inner.id in files:
                    files.setdefault(node.targets[0].id, files[inner.id])

    checks = next(node.value for node in ast.walk(function)
                  if isinstance(node, ast.Assign)
                  and any(getattr(t, "id", "") == "checks" for t in node.targets))
    dependencies: dict[str, set[str]] = {}
    for key, value in zip(checks.keys, checks.values):
        used = {files[n.id] for n in ast.walk(value)
                if isinstance(n, ast.Name) and n.id in files}
        dependencies[key.value] = used
    return dependencies


def _red_checks(evidence_dir, removed=None):
    if removed is not None:
        (evidence_dir / removed).unlink()
    try:
        gate.verify(evidence_dir / ARCHITECTURE)
        return set()
    except SystemExit as exit_error:
        return set(str(exit_error).splitlines()[0].split(": ", 1)[1].split(", "))


def test_removing_an_evidence_file_reddens_exactly_the_checks_that_read_it(evidence_dir):
    dependencies = _evidence_dependencies()
    baseline_red = _red_checks(evidence_dir)
    present = {path.name for path in evidence_dir.iterdir()}
    covered = sorted({f for files in dependencies.values() for f in files} & present)
    assert covered, "no evidence file is both present and referenced"

    mismatches = {}
    for filename in covered:
        backup = (evidence_dir / filename).read_bytes()
        newly_red = _red_checks(evidence_dir, removed=filename) - baseline_red
        (evidence_dir / filename).write_bytes(backup)
        expected = {name for name, files in dependencies.items()
                    if filename in files} - baseline_red
        if newly_red != expected:
            mismatches[filename] = {"went_red": sorted(newly_red),
                                    "reads_it": sorted(expected)}
    assert mismatches == {}
