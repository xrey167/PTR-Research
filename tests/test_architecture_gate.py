"""Tests for the gate's evidence handling.

The gate compares recorded numbers; these tests cover how it behaves when a
recorded file is absent, which used to be indistinguishable from a threshold
violation and in one case let a check pass with no evidence at all.
"""
import json
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


def _gate_report(evidence_dir):
    """The gate's structured report, red or green.

    A refused gate carries it on the SystemExit. Before that, a test that
    wanted one of these fields on a red tree had to recompute the gate's own
    logic — and a reimplementation is not a check of the thing it
    reimplements. This repository has that defect on record twice; this is
    how it stops being available.
    """
    try:
        return gate.verify(evidence_dir / ARCHITECTURE)
    except SystemExit as refused:
        report = getattr(refused, "report", None)
        assert report is not None, "a refused gate must carry its report"
        return report


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


# ---------------------------------------------------------------------------
# The producer-stamp ratchet. Unstamped evidence fails the gate; the
# grandfather set is the list of files recorded before research/evidence.py
# existed. It is the last hand-kept list in the gate, so it gets the same
# treatment as `baseline_of`: a test that fails when it drifts.


def test_the_grandfather_set_only_names_files_that_are_actually_unstamped(evidence_dir):
    """A name that stays in the set after its benchmark was re-run through
    research.evidence.write() would silently exempt a file that no longer
    needs exempting — and the next unstamped recording of it would pass."""
    still_needed = []
    for name in sorted(gate.UNSTAMPED_GRANDFATHERED):
        path = evidence_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("producer"):
            still_needed.append(name)
    stamped_but_exempt = [
        name for name in sorted(gate.UNSTAMPED_GRANDFATHERED)
        if (evidence_dir / name).exists()
        and json.loads((evidence_dir / name).read_text(encoding="utf-8")).get("producer")
    ]
    assert stamped_but_exempt == [], (
        "these files carry a producer stamp and must be removed from "
        "UNSTAMPED_GRANDFATHERED: " + ", ".join(stamped_but_exempt))
    assert still_needed, "the grandfather set has gone empty - delete it"


def test_unstamped_evidence_outside_the_grandfather_set_fails_the_gate(evidence_dir):
    """The ratchet: the debt may be paid off, never taken on again."""
    stamped = next(
        path for path in evidence_dir.iterdir()
        if path.suffix == ".json"
        and path.name not in gate.UNSTAMPED_GRANDFATHERED
        and json.loads(path.read_text(encoding="utf-8")).get("producer"))
    backup = stamped.read_bytes()
    data = json.loads(backup.decode("utf-8"))
    data.pop("producer")
    data.pop("producer_sha256", None)
    stamped.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        with pytest.raises(SystemExit) as failure:
            gate.verify(evidence_dir / ARCHITECTURE)
        assert "without a producer stamp" in str(failure.value)
        assert stamped.name in str(failure.value)
    finally:
        stamped.write_bytes(backup)


def test_every_evidence_file_the_gate_reads_goes_through_the_staleness_pass(evidence_dir):
    """The staleness pass used to iterate a hand-kept 8-tuple while the gate
    read 37 files, so 30 of them were exempt by omission. It now iterates
    what `_read` registered, which cannot fall behind."""
    try:
        gate.verify(evidence_dir / ARCHITECTURE)
    except SystemExit:
        pass
    registered = set(gate._LOADED)
    dependencies = _evidence_dependencies()
    read_by_a_check = {f for files in dependencies.values() for f in files
                       if f.endswith(".json")}
    assert read_by_a_check, "no JSON evidence is referenced by any check"
    assert read_by_a_check <= registered, (
        "read by a check but never registered for the staleness pass: "
        + ", ".join(sorted(read_by_a_check - registered)))


def test_changing_the_measured_code_makes_its_evidence_stale(evidence_dir, tmp_path):
    """The drift the producer stamp could not see.

    Evidence bound only to its producer stays green while the SYSTEM changes
    underneath it. Measured before this existed: replacing any of seven core
    modules with one that raises on import turned zero of forty-five checks
    red. A file that names a `subject` now goes stale the moment that subject
    does, which is the property the gate is supposed to have.
    """
    subject_file = REPO / "neural_pods" / "storage.py"
    # Named, not "whichever file happens to carry a subject": picking by
    # filter meant that if the storage evidence stopped naming storage.py,
    # another record would be selected and the test would still pass.
    stamped = evidence_dir / "storage-facade-20260920.json"
    data = json.loads(stamped.read_text(encoding="utf-8"))
    # `x in data["subject"] or data["subject"]` stood here. Python reads that
    # as `(x in list) or (list)`, and a non-empty list is truthy — so the
    # assertion held whether or not the path was present. The same defect
    # this PR removed from the benchmarks, written into the test that checks
    # the removal.
    assert subject_file.relative_to(REPO).as_posix() in data["subject"]

    # Not by editing the repository: by recomputing the stamp against a tree
    # in which the measured module differs.
    import evidence as evidence_module

    original = data["subject_sha256"]
    moved_on = dict(data)
    fake_root = tmp_path / "tree"
    for relative in data["subject"]:
        target = fake_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("raise RuntimeError('gutted')\n", encoding="utf-8")
    moved_on["subject_sha256"] = evidence_module.subject_sha256(
        data["subject"], project_root=fake_root)
    assert moved_on["subject_sha256"] != original

    backup = stamped.read_bytes()
    stamped.write_text(json.dumps(moved_on, indent=2), encoding="utf-8")
    try:
        with pytest.raises(SystemExit) as failure:
            gate.verify(evidence_dir / ARCHITECTURE)
        assert "the code it measured has changed" in str(failure.value)
        assert stamped.name in str(failure.value)
    finally:
        stamped.write_bytes(backup)


def test_evidence_that_names_no_subject_is_reported_not_hidden(evidence_dir):
    """A file bound to its producer alone is a known gap. It is named in the
    gate's own output so it cannot quietly become the normal case again.

    This test used to recompute the list itself in the `except SystemExit`
    branch — and since the gate is red on this tree, that branch always ran,
    so `evidence_without_a_subject` was never read at all. Its closing
    assertion then compared two sets both derived from `gate._LOADED` by the
    test, one requiring `subject` and the other requiring its absence: they
    are disjoint by construction and the assertion could not fail.

    It now asserts a property of the GATE's own output, on a run whose result
    it actually obtains.
    """
    reported = _gate_report(evidence_dir)["evidence_without_a_subject"]

    # Everything the gate names must be a file that really lacks a subject...
    for name in reported:
        data = json.loads((evidence_dir / name).read_text(encoding="utf-8"))
        assert data.get("producer"), f"{name} is not even stamped"
        assert not data.get("subject"), f"{name} does name a subject"

    # ...and every stamped file the gate READ without one must be named.
    # Only files a check actually reads are in scope: the gate cannot report
    # on evidence it never opened, and research/runs/ holds more than that.
    expected = []
    for name in gate._LOADED:
        data = gate._LOADED[name]
        if isinstance(data, dict) and data.get("producer") and not data.get("subject"):
            expected.append(name)
    assert reported == sorted(expected)


def test_the_subject_report_goes_red_when_a_subject_is_dropped(evidence_dir):
    """The direction that matters: remove a subject and the gate must start
    naming that file. Without this, the test above passes on a gate that
    reports an empty list forever."""
    stamped = evidence_dir / "storage-facade-20260920.json"
    backup = stamped.read_bytes()
    assert stamped.name not in _gate_report(evidence_dir)["evidence_without_a_subject"]

    data = json.loads(backup.decode("utf-8"))
    data.pop("subject")
    data.pop("subject_sha256", None)
    stamped.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        assert stamped.name in _gate_report(evidence_dir)["evidence_without_a_subject"]
    finally:
        stamped.write_bytes(backup)


# ---------------------------------------------------------------------------
# The `tests` check is the only one that can execute something, which makes
# its own fail-open modes the most expensive ones in the gate.


@pytest.mark.parametrize("summary,expected", [
    ("447 passed, 8 skipped in 35.50s",
     {"passed": 447, "failed": 0, "errors": 0, "skipped": 8}),
    # The line that used to parse to failed 0 and go through.
    ("300 passed, 4 errors in 5s",
     {"passed": 300, "failed": 0, "errors": 4, "skipped": 0}),
    ("2 failed, 299 passed, 31 errors in 9s",
     {"passed": 299, "failed": 2, "errors": 31, "skipped": 0}),
    ("1 failed, 10 passed, 3 skipped in 1s",
     {"passed": 10, "failed": 1, "errors": 0, "skipped": 3}),
])
def test_every_pytest_count_is_parsed_independently(summary, expected):
    """pytest prints the counts in a varying order and omits the zeros. One
    positional pattern is how `4 errors` was read as `failed: 0`."""
    import record_test_run

    counts = {}
    for name, pattern in record_test_run.COUNT.items():
        found = pattern.search(summary)
        counts[name] = int(found.group(1)) if found else 0
    for key, value in expected.items():
        assert counts[key] == value, f"{key} in {summary!r}"


@pytest.mark.parametrize("run,reason", [
    ({"exit_code": 1, "failed": 0, "errors": 0, "passed": 400, "skipped": 0},
     "a non-zero exit code"),
    ({"exit_code": 0, "failed": 0, "errors": 4, "passed": 400, "skipped": 0},
     "pytest errors"),
    ({"exit_code": 0, "failed": 0, "errors": 0, "passed": 400, "skipped": 99},
     "an unbounded number of skips"),
])
def test_the_tests_check_rejects_a_run_that_only_looks_clean(evidence_dir, run, reason):
    """Each of these used to pass: the check read `failed` and `passed` and
    nothing else, so a run that errored out, or that skipped most of itself,
    was indistinguishable from a green one."""
    import record_test_run

    recorded = dict(run)
    recorded["sources_sha256"] = record_test_run.source_fingerprint(REPO)
    (evidence_dir / "tests-20260920.json").write_text(
        json.dumps(recorded, indent=2), encoding="utf-8")

    red = _red_checks(evidence_dir)
    assert "tests" in red, f"the gate accepted a run with {reason}"
