"""Make the repository importable for the whole suite.

Tests reach into `research/` for the gate, the generators and the benchmark
cores. Every file used to prepend its own sys.path entry, so the import
layout was restated a dozen times and each copy silently depended on that
file's depth in the tree.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

for entry in (REPO, REPO / "research"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
