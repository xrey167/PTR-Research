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
  2. `assert <x> in C or C` — a membership test disjoined with its own
     container. This is the shape that was actually written here: the
     disjunction is true whenever `c` is truthy, and when `c` is empty
     `x in c` is False too. The whole assertion reduces to `bool(c)`, and the
     membership test decides nothing. `not in` is included, for the same
     reason with the cases swapped.

     It is this shape and not "the right operand appears somewhere inside the
     left", which is what the rule asked first: `transform(value) or value`
     satisfies that reading and can fail, when both sides are falsy.

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


def _membership_over_its_own_container(values: list[ast.expr]) -> bool:
    """Whether the chain is `x in c or c` — a test of `c` against itself.

    `c` truthy makes the disjunction true; `c` empty makes `x in c` false as
    well, so the whole assertion is `bool(c)` and the membership test decides
    nothing. `x not in c or c` is the same: an empty `c` satisfies the left
    operand instead of the right.

    This checks that exact shape and no other. The first version of the rule
    asked whether any later operand appeared ANYWHERE inside an earlier one,
    which is not the same claim: `transform(value) or value` matches it and
    can fail perfectly well, when both sides are falsy. A rule that flags
    honest code is switched off within a week, so it catches less rather than
    guessing more.
    """
    for index, value in enumerate(values[:-1]):
        if not (isinstance(value, ast.Compare) and len(value.ops) == 1
                and isinstance(value.ops[0], (ast.In, ast.NotIn))):
            continue
        container = ast.unparse(value.comparators[0])
        if container in {ast.unparse(later) for later in values[index + 1:]}:
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
                or _membership_over_its_own_container(test.values):
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

    # The shape from the real defect, spelled out: a membership test
    # disjoined with the container it tests, so the left decides nothing.
    redundant = ast.parse('assert name in report["names"] or report["names"]\n')
    assert _vacuous_disjuncts(redundant)

    # Same shape with the two cases swapped: an empty container satisfies the
    # left operand instead of the right.
    negated = ast.parse('assert name not in report["names"] or report["names"]\n')
    assert _vacuous_disjuncts(negated)

    # And with something between the two operands, which changes nothing.
    spaced = ast.parse('assert key in mapping or ready or mapping\n')
    assert _vacuous_disjuncts(spaced)


def test_the_rule_leaves_legitimate_disjunctions_alone():
    """The other half: a rule that flags honest code gets switched off."""
    for code in (
        'assert a or b\n',
        'assert value is None or bool(value)\n',
        'assert "x" in items or "y" in items\n',
        'assert report or not expected\n',
        'assert flag or []\n',              # empty container: falsy, fine
        'assert flag or ""\n',
        # Two independent names: both can be false at the same time.
        'assert found or expected\n',
        'assert items[0] or items[1]\n',
        # The false positive the first version of the rule produced. Both
        # sides can be falsy at once, so this assertion can fail.
        'assert transform(value) or value\n',
        'assert normalise(name) or name\n',
        'assert compute(rows)[0] or rows\n',
        # A membership test against a DIFFERENT container decides something.
        'assert name in expected or names\n',
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
