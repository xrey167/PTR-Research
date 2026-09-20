"""Materialize the fresh holdout evaluation split and its manifest.

Writes both under research/runs/ (checked in), so a clone can regenerate the
split and compare it byte for byte. The manifest records the sha256 of the
generator itself: the frozen dev/test splits lost that property — they hold
132 cases while today's reader_training_data.build_data() produces 96, so the
code that made them is no longer in the repository and they cannot be
rebuilt. This split is not allowed to drift the same way.

Usage:  python research/generate_holdout_split.py [--check]
        --check regenerates in memory and fails if the checked-in artifact
        differs; that is what the test and the gate rely on.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.reader_holdout_data import (  # noqa: E402
    build_holdout, disjointness_report)

RUNS = Path(__file__).resolve().parent / "runs"
SPLIT_FILE = RUNS / "reader-holdout-split-20260920.json"
MANIFEST_FILE = RUNS / "reader-holdout-manifest-20260920.json"
GENERATOR = Path(__file__).resolve().parent / "reader_holdout_data.py"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render() -> tuple[str, dict]:
    rows = build_holdout()
    payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    report = disjointness_report()
    families: dict[str, int] = {}
    for row in rows:
        families[row["task"]] = families.get(row["task"], 0) + 1
    manifest = {
        "schema": "reader-holdout-split:v1",
        "split": "holdout",
        "purpose": ("uncontaminated evaluation split; the frozen test split "
                    "saturated at 125/132 raw and 44/44 concept, so it can no "
                    "longer separate one reader generation from the next"),
        "rows": len(rows),
        "families": dict(sorted(families.items())),
        "languages": sorted({row["language"] for row in rows}),
        "cases_sha256": _sha256_text(payload),
        "generator": GENERATOR.name,
        "generator_sha256": _sha256_text(GENERATOR.read_text(encoding="utf-8")),
        "disjoint_from_frozen_splits": report["disjoint"],
        "overlaps": report["overlaps"],
        "frozen_rows_compared": report["frozen_rows"],
        "scope": ("generation only - no model was run against this split here; "
                  "evaluating it needs the reader checkpoint on the server"),
    }
    return payload, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the checked-in artifact matches the generator")
    args = parser.parse_args()
    payload, manifest = render()

    if args.check:
        problems = []
        if not SPLIT_FILE.exists():
            problems.append(f"missing {SPLIT_FILE.name}")
        elif SPLIT_FILE.read_text(encoding="utf-8") != payload:
            problems.append(f"{SPLIT_FILE.name} differs from the generator output")
        if not MANIFEST_FILE.exists():
            problems.append(f"missing {MANIFEST_FILE.name}")
        else:
            stored = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            for key in ("cases_sha256", "generator_sha256", "rows"):
                if stored.get(key) != manifest[key]:
                    problems.append(f"manifest {key} is stale")
        if not manifest["disjoint_from_frozen_splits"]:
            problems.append(f"split overlaps the frozen splits: {manifest['overlaps']}")
        if problems:
            raise SystemExit("holdout split check failed: " + "; ".join(problems))
        print(json.dumps({"ok": True, "rows": manifest["rows"],
                          "cases_sha256": manifest["cases_sha256"]}, indent=2))
        return

    RUNS.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(payload, encoding="utf-8")
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "rows", "families", "cases_sha256", "disjoint_from_frozen_splits")}, indent=2))


if __name__ == "__main__":
    main()
