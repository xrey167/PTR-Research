"""NeoHorse reference manifest and the local GPU probe it points at.

Both artifacts are produced on the server: the manifest transcribes the
published ten-benchmark table, and local_probe points at a real generation
throughput measurement on the checkpoint. Neither can be synthesised here,
so this test skips when they are absent rather than failing on a bare
FileNotFoundError - and names what is missing, so committing the manifest
under research/runs/ is all it takes to make it run again.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = "neohorse-reference-manifest-001.json"
CANDIDATES = [REPO / "research" / "runs" / MANIFEST, REPO / "runs" / MANIFEST]


def _manifest_path():
    return next((c for c in CANDIDATES if c.exists()), None)


pytestmark = pytest.mark.skipif(
    _manifest_path() is None,
    reason=f"{MANIFEST} is a server artifact and is not in this checkout; "
           "commit it under research/runs/ to enable this test")


def test_neohorse_reference_manifest_and_probe():
    path = _manifest_path()
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m['model'] == 'TokenRhythm/NeoHorse-1-4B'
    assert m['published_macro_average']['delta'] == 5.93
    assert len(m['published_benchmarks']) == 10
    probe_path = Path(m['local_probe'])
    if not probe_path.is_absolute():
        probe_path = next((base / probe_path for base in (REPO, path.parent)
                           if (base / probe_path).exists()), probe_path)
    if not probe_path.exists():
        pytest.skip(f"local probe {m['local_probe']} is not in this checkout")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    assert probe['schema'] == 'neohorse-real-checkpoint-probe:v1'
    assert probe['total_generated_tokens'] > 0
    assert probe['mean_tokens_per_s'] > 0
