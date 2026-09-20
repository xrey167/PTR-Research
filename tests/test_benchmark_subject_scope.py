"""Every benchmark that stamps a subject must define it at module scope.

The defect this pins was mine, and the mechanism is worth recording. A helper
script inserted `SUBJECT` after "the last top-level import", found by
`line.startswith(("import ", "from "))` over the raw lines — which matched an
import line INSIDE an f-string holding the code for a peer process. `SUBJECT`
ended up in the string. The module had no binding, so `write_evidence(...,
subject=SUBJECT)` raised NameError before any evidence was written.

Nothing caught it because `research/benchmark_mesh_cache.py` needs LXD nodes
and a Redis server to reach that line — the untested `collect()` half named
in the report. A single AST check covers every benchmark, including the ones
that can only run on the server.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

RESEARCH = Path(__file__).resolve().parents[1] / "research"
BENCHMARKS = sorted(RESEARCH.glob("benchmark_*.py"))


def _module_level_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _stamping_benchmarks():
    for path in BENCHMARKS:
        try:
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source)
        except (UnicodeDecodeError, SyntaxError):
            continue
        uses_subject = any(
            isinstance(node, ast.Call)
            and any(kw.arg == "subject" for kw in node.keywords)
            for node in ast.walk(tree))
        if uses_subject:
            yield path, tree


@pytest.mark.parametrize("path,tree", list(_stamping_benchmarks()),
                         ids=lambda value: getattr(value, "name", ""))
def test_a_stamped_benchmark_binds_its_subject_at_module_scope(path, tree):
    names = _module_level_names(tree)
    referenced = {
        kw.value.id
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "subject" and isinstance(kw.value, ast.Name)}
    missing = sorted(referenced - names)
    assert not missing, (
        f"{path.name}: passes subject={missing} to write_evidence(), but "
        f"does not define it at module scope — the call raises NameError "
        f"before any evidence is written")


def test_the_scan_found_the_benchmarks():
    """An empty parametrisation would make every case above vacuous."""
    stamping = list(_stamping_benchmarks())
    assert len(stamping) >= 10
    names = {path.name for path, _ in stamping}
    assert "benchmark_mesh_cache.py" in names


def test_a_subject_hidden_in_a_string_is_not_module_scope():
    """The counter-check, in the exact shape the defect had: the assignment
    is inside a string that holds code for another process."""
    source = "\n".join([
        'peer_code = """',
        'from neural_pods.mesh_cache import MeshCache',
        'SUBJECT = ["neural_pods/mesh_cache.py"]',
        '"""',
        'write_evidence(result, OUT, __file__, subject=SUBJECT)',
    ])
    tree = ast.parse(source)
    assert "SUBJECT" not in _module_level_names(tree)
    assert "peer_code" in _module_level_names(tree)
