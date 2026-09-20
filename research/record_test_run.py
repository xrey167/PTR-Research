"""Run the test suite and record the result as gate evidence.

The gate's `tests` check used to read a passed-count out of
architecture-20260917.json, a file written days before the code it claimed
to cover. Nothing tied the number to a revision, so a regression introduced
after the recording was invisible to a "fail-closed" gate.

The report written here carries `sources_sha256`, a digest over every .py
file under neural_pods/, research/ and tests/. The gate recomputes it and
refuses evidence that was produced by different code. Re-record after every
change:

    python research/record_test_run.py

`--check` re-runs and fails if the stored report no longer matches, without
overwriting it.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "research" / "runs" / "tests-20260920.json"
SOURCE_DIRS = ("neural_pods", "research", "tests")
SUMMARY = re.compile(
    r"(?:(?P<failed>\d+) failed)?,?\s*(?P<passed>\d+) passed"
    r"(?:,\s*(?P<skipped>\d+) skipped)?")


def source_fingerprint(project_root: Path = PROJECT) -> str:
    """Digest over the code whose behaviour a test run describes."""
    import hashlib
    digest = hashlib.sha256()
    for directory in SOURCE_DIRS:
        base = project_root / directory
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(project_root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_pytest(project_root: Path = PROJECT) -> dict:
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    completed = subprocess.run(command, cwd=project_root, capture_output=True,
                               text=True)
    tail = completed.stdout.strip().splitlines()
    summary_line = next((line for line in reversed(tail)
                         if " passed" in line or " failed" in line), "")
    match = SUMMARY.search(summary_line)
    counts = {key: int(match.group(key) or 0) for key in ("passed", "failed", "skipped")} \
        if match else {"passed": 0, "failed": -1, "skipped": 0}
    return {
        "schema": "pytest-run:v1",
        "command": " ".join(command[1:]),
        "exit_code": completed.returncode,
        "summary_line": summary_line,
        **counts,
        "python": platform.python_version(),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sources_sha256": source_fingerprint(project_root),
        "source_dirs": list(SOURCE_DIRS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the stored report without overwriting it")
    args = parser.parse_args()
    result = run_pytest()

    if args.check:
        if not OUT.exists():
            raise SystemExit(f"no recorded test run at {OUT}")
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        problems = [f"{key}: stored {stored.get(key)} vs now {result[key]}"
                    for key in ("passed", "failed", "sources_sha256")
                    if stored.get(key) != result[key]]
        if problems:
            raise SystemExit("recorded test run is stale: " + "; ".join(problems))
        print(json.dumps({"ok": True, **{k: stored[k] for k in
                                         ("passed", "failed", "skipped")}}, indent=2))
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
