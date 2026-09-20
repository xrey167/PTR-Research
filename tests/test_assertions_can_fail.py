"""Every assertion in the suite must be capable of failing.

The defect this pins was found twice in this repository's own tests, both
times written while removing the same defect from the benchmarks:

    assert str(path) in data["subject"] or data["subject"]

Python reads that as `(x in list) or (list)`. A non-empty list is truthy, so
the assertion held whether or not the path was there. `quoted_key_roundtrip:
True` and `reflex_frames_valid == 132` were the same mistake one layer out —
a claim whose truth does not depend on what is being claimed.

A test that cannot fail is worse than a missing test: it produces the
confidence without performing the check, and nothing downstream can tell the
difference.

Deliberately narrow. It looks for two syntactic shapes and nothing else:

  1. `assert <anything> or <literal>` where the literal is truthy.
  2. `assert <expr containing C> or C` — the right operand appears inside the
     left. This is the shape that was actually written here: in
     `x in c or c`, the disjunction is true whenever `c` is truthy, and when
     `c` is empty `x in c` is False too. The whole assertion reduces to
     `bool(c)`, and the membership test decides nothing.

A broad heuristic over "suspicious assertions" would flag the many legitimate
compound conditions in this suite and be switched off within a week. A rule
that survives is worth more than a rule that catches everything once.

Sources are read with `utf-8-sig`: eight files in this repository carry a
UTF-8 BOM, and `ast.parse` rejects it. Reading them as plain utf-8 turned
this scan into a scan of the files that happened to parse — which is the
same class of defect the file is about.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SOURCES = sorted(TESTS.glob("test_*.py"))


def _always_true(node: ast.AST) -> bool:
    """Whether this expression is truthy no matter what the program does."""
    if isinstance(node, ast.Constant):
        return bool(node.value)
    # A non-empty literal container is truthy by construction.
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.keys)
    return False


def _subexpressions(node: ast.AST) -> set[str]:
    """Every sub-expression of `node`, as normalised source."""
    return {ast.unparse(inner) for inner in ast.walk(node)
            if isinstance(inner, ast.expr)}


def _redundant_operand(values: list[ast.expr]) -> bool:
    """Whether a later operand already appears inside an earlier one.

    `x in c or c`: the disjunction is true whenever `c` is truthy, and when
    `c` is empty the membership test is False as well. The assertion says
    `bool(c)` and nothing more.
    """
    for index, value in enumerate(values):
        text = ast.unparse(value)
        for earlier in values[:index]:
            if text in _subexpressions(earlier) - {ast.unparse(earlier)}:
                return True
    return False


def _vacuous_disjuncts(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, source) for every `assert a or b` that cannot be false."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or)):
            continue
        # An always-true operand anywhere makes the whole chain unconditional.
        if any(_always_true(value) for value in test.values) \
                or _redundant_operand(test.values):
            found.append((node.lineno, ast.unparse(test)))
    return found


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_no_assertion_is_unconditionally_true(source):
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    vacuous = _vacuous_disjuncts(tree)
    assert not vacuous, (
        f"{source.name}: assertion(s) that cannot fail — "
        + "; ".join(f"line {line}: {text}" for line, text in vacuous))


def test_the_rule_catches_the_shape_it_was_written_for():
    """The counter-check. A rule that has never been seen to fail is not a
    rule — this is the exact line that stood in test_architecture_gate.py."""
    offending = ast.parse(
        'assert str(subject_file) in data["subject"] or data["subject"]\n')
    assert _vacuous_disjuncts(offending)

    literal = ast.parse('assert value == 1 or True\n')
    assert _vacuous_disjuncts(literal)

    container = ast.parse('assert found or ["fallback"]\n')
    assert _vacuous_disjuncts(container)

    # The shape from the real defect, spelled out: the right operand is a
    # sub-expression of the left, so the left decides nothing.
    redundant = ast.parse('assert name in report["names"] or report["names"]\n')
    assert _vacuous_disjuncts(redundant)


def test_the_rule_leaves_legitimate_disjunctions_alone():
    """The other half: a rule that flags honest code gets switched off."""
    for code in (
        'assert a or b\n',
        'assert value is None or bool(value)\n',
        'assert "x" in items or "y" in items\n',
        'assert report or not expected\n',
        'assert flag or []\n',              # empty container: falsy, fine
        'assert flag or ""\n',
        # Same NAME on both sides is not the shape: neither is a
        # sub-expression of the other, and both can be false.
        'assert found or expected\n',
        'assert items[0] or items[1]\n',
    ):
        assert not _vacuous_disjuncts(ast.parse(code)), code


def test_the_suite_is_actually_being_scanned():
    """Without this, an empty glob would make every case above vacuous —
    which would be a fitting way to fail at exactly this task."""
    assert len(SOURCES) > 30
    assert any(path.name == "test_architecture_gate.py" for path in SOURCES)
    # And every one of them must actually parse. A file the scan silently
    # skipped would be a file this rule does not cover.
    for source in SOURCES:
        ast.parse(source.read_text(encoding="utf-8-sig"))
