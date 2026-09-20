"""Stamp a benchmark's output with the script that produced it AND with the
code it measured.

Two different gaps, and the second one is the wider.

THE PRODUCER GAP. A benchmark gets changed, its recorded evidence stays, and
nothing notices that the numbers in research/runs/ can no longer be produced
by the code in the repository. That is the defect the 2026-09-20 review found
in the frozen evaluation splits (132 recorded cases, 96 the generator makes
today) and then again in taskgraph, traced-pipeline and perception evidence.
`producer` + `producer_sha256` close it.

THE SUBJECT GAP. Evidence binds to its producer, never to what it is evidence
ABOUT. So the more common drift stayed invisible: the SYSTEM changes, the
benchmark does not, the recorded numbers describe code that no longer exists.
Measured rather than argued: replacing neural_pods/mesh.py, taskgraph.py,
storage.py, dream.py, native_comm.py, mesh_cache.py or perception.py with a
module that raises on import turned NO gate check red. Forty-four of
forty-five stayed green while the thing they are evidence about was gone.

`subject` names the modules a benchmark exercises and `subject_sha256` is a
digest over them, built the same way `record_test_run.source_fingerprint()`
builds its digest over the tested tree — the one construction in this
repository that already binds evidence to its subject. The gate recomputes
both hashes and refuses evidence whose producer or whose subject has moved
on. A benchmark that names no subject is accepted, and says so in the
report, because an un-named subject is a known gap rather than a hidden one.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

RESEARCH = Path(__file__).resolve().parent
PROJECT = RESEARCH.parent


def script_sha256(script: str | Path) -> str:
    return hashlib.sha256(Path(script).read_bytes()).hexdigest()


def subject_sha256(subject: Iterable[str], *, project_root: Path = PROJECT) -> str:
    """Digest over the source files a benchmark's numbers describe.

    Path and content, in sorted order, exactly like
    `record_test_run.source_fingerprint()`. A missing file is hashed as such
    rather than skipped: a module that was deleted has to change the digest,
    otherwise the check would ignore the most drastic change there is.
    """
    digest = hashlib.sha256()
    for relative in sorted(subject):
        path = project_root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def stamp(result: dict[str, Any], producer: str | Path,
          *, subject: Sequence[str] | None = None,
          project_root: Path = PROJECT) -> dict[str, Any]:
    """Add the producing script and the measured modules to a result."""
    path = Path(producer).resolve()
    result["producer"] = path.name
    result["producer_sha256"] = script_sha256(path)
    if subject is not None:
        result["subject"] = sorted(subject)
        result["subject_sha256"] = subject_sha256(subject,
                                                  project_root=project_root)
    return result


def write(result: dict[str, Any], out: str | Path, producer: str | Path,
          *, subject: Sequence[str] | None = None, default=None,
          project_root: Path = PROJECT) -> dict[str, Any]:
    """Stamp and write a benchmark result in one step."""
    stamp(result, producer, subject=subject, project_root=project_root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=default) + "\n",
                   encoding="utf-8")
    return result
