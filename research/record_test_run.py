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

WHAT THE SUMMARY LINE DOES NOT SAY, and why the counts are no longer read
out of it alone. pytest reports collection and fixture failures as `errors`,
not as `failed`: a run ending `300 passed, 4 errors` parsed to
`failed: 0, passed: 300` and went through the gate. The same line said
nothing about how many tests were skipped, so a suite that quietly stopped
running half of itself kept the check green.

`errors` is parsed, `skipped` is bounded, and pytest's EXIT CODE — the one
value that is structured rather than reconstructed from text — is what the
gate reads first. A non-zero exit code fails the check whatever the numbers
say.
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
#: Each count is matched independently, because pytest prints them in a
#: varying order and omits the ones that are zero. A single positional
#: pattern is how `4 errors` came to be read as `failed: 0`.
COUNT = {name: re.compile(rf"(\d+) {name}")
         for name in ("failed", "passed", "skipped", "errors", "xfailed",
                      "xpassed")}


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
    counts = {}
    for name, pattern in COUNT.items():
        found = pattern.search(summary_line)
        counts[name] = int(found.group(1)) if found else 0
    if not summary_line:
        # No summary at all means the run did not get far enough to produce
        # one. Recording zeros here would look like a clean slate.
        counts["failed"] = -1
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
