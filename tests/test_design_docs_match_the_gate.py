"""A design document may not mark a gate check as done that the gate refuses.

This is the mistake this project has made more often than any other. The
phase tables in the design documents carry a gate-check name per phase, and a
name with a tick beside it reads as "measured, verified, done". Three of them
were not:

  * `dream_reflex ✔` while the check was red — the benchmark had fallen back
    to a synthetic two-generation pool.
  * `xgboost_pod`, `perception_stream`, `improve_cycle`, `latent_addressing`
    listed as the checks for phases P2–P5; the first did not exist and the
    other three still do not.
  * `reflex_dispatch grün` while the recorded evidence said the reflex
    resolved nothing at all.

Every one was found by a person reading carefully. This is that reading, as
a check: the tick and the gate have to agree, and when they disagree the
document is wrong, because the gate is the thing that decides promotion.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import verify_architecture_gate as gate

REPO = Path(__file__).resolve().parents[1]
DESIGN_DOCS = sorted(
    list((REPO / "research").glob("*DESIGN*.md"))
    + [REPO / "ARCHITECTURE-MASTER-20260920.md"])

#: `check_name` followed by a tick, in a table cell or prose.
TICKED = re.compile(r"`([a-z_][a-z0-9_]*)`\s*(?:\*\*)?✔")


def _check_names() -> set[str]:
    """Every check the gate defines, read from its source."""
    tree = ast.parse((REPO / "research" / "verify_architecture_gate.py")
                     .read_text(encoding="utf-8"))
    verify = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "verify")
    checks = next(node.value for node in ast.walk(verify)
                  if isinstance(node, ast.Assign)
                  and any(getattr(t, "id", "") == "checks" for t in node.targets))
    return {key.value for key in checks.keys}


def _gate_checks() -> dict[str, bool]:
    try:
        return gate.verify(REPO / "research" / "runs" / "architecture-20260917.json")["checks"]
    except SystemExit as refused:
        report = getattr(refused, "report", None)
        assert report is not None, "a refused gate must carry its report"
        return report["checks"]


@pytest.mark.parametrize("document", DESIGN_DOCS, ids=lambda p: p.name)
def test_no_document_ticks_a_check_the_gate_refuses(document):
    checks = _gate_checks()
    text = document.read_text(encoding="utf-8-sig")

    wrong = []
    for line in text.splitlines():
        for name in TICKED.findall(line):
            if name in checks and not checks[name]:
                wrong.append((name, line.strip()[:110]))
    assert not wrong, (
        f"{document.name}: marks a red gate check as done — "
        + "; ".join(f"`{name}` in: {line}" for name, line in wrong))


#: A backticked name, as a phase table's gate-check cell writes them: one
#: name, or several separated by `/`, `,` or `+`.
NAME = re.compile(r"`([a-z_][a-z0-9_]*)`")

#: Prose that presents what follows as the gate check for something. The
#: anchor is the words themselves, so a name only counts when the document
#: called it a gate check.
GATE_CHECK = re.compile(r"[Gg]ate[-‑ ]?[Cc]hecks?\b")

#: How far past the anchor a name still belongs to it.
PROSE_WINDOW = 120

#: Markers that turn a check name from a claim into a statement of fact.
HONEST_MARKERS = (
    "existiert nicht", "does not exist", "fehlt", "(offen)", "offen)",
    "nicht umgesetzt", "noch nicht", "rot", "geplant",
)


def _is_separator(line: str) -> bool:
    """`|---|---|`, the row that marks the line above it as a header."""
    return (line.startswith("|") and "-" in line
            and set(line) <= set("|-: "))


def _gate_check_claims(text: str) -> list[tuple[str, str]]:
    """Every (name, context) a document presents AS a gate check.

    Two places do that, and both are read here rather than guessed at:

      * the gate-check column of a phase table, found by its own header, and
      * prose that says "Gate-Check" and then names one.

    A backticked name anywhere else — a module, a parameter, a metric in a
    neighbouring column — is not a claim about the gate and is not collected.
    That distinction is the whole rule. It used to be drawn by a hand-kept
    set of names already known to be invented, which let every NEW invention
    straight through.
    """
    claims: list[tuple[str, str]] = []

    # Phase tables. A header row is one the separator row follows — without
    # that, a data row whose first cell reads "Gate-Checks definiert" is
    # mistaken for a header and the wrong column gets read.
    lines = text.splitlines()
    column: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|"):
            column = None
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if _is_separator(following):
            column = next((i for i, cell in enumerate(cells)
                           if GATE_CHECK.search(cell)), None)
            continue
        if column is None or _is_separator(stripped):
            continue
        if column < len(cells):
            # The WHOLE row is the context: an honest marker sits wherever
            # the sentence needed it, and a truncated context turned
            # "`latent_addressing` **existiert nicht**" into a claim.
            claims.extend((name, stripped)
                          for name in NAME.findall(cells[column]))

    # Prose. Newlines become spaces so a claim wrapped across two lines is
    # still one claim; a cell or sentence boundary ends the window, because
    # the next cell is about something else.
    flat = text.replace("\n", " ")
    for match in GATE_CHECK.finditer(flat):
        window = re.split(r"\||\.\s", flat[match.end():match.end() + PROSE_WINDOW])[0]
        context = flat[max(0, match.start() - 40):match.end() + 80]
        claims.extend((name, context.strip()) for name in NAME.findall(window))
    return claims


@pytest.mark.parametrize("document", DESIGN_DOCS, ids=lambda p: p.name)
def test_no_document_names_a_gate_check_that_does_not_exist(document):
    """The other half. A phase table listing `improve_cycle` as its gate
    check, when no such check is defined, reads as a verified phase and is
    not one. Naming it as missing is fine; presenting it as a check is not.

    This rule is now the inverse of what it was. It used to flag only names
    on a hand-kept list of inventions already found — so a name invented
    tomorrow passed, and the test written to close that class of mistake
    reproduced it. Every name a document presents as a gate check must be one
    the gate defines.
    """
    defined = _check_names()
    text = document.read_text(encoding="utf-8-sig")

    # A name presented WITH an honest marker is not a claim. The markers are
    # explicit and few on purpose: "(offen)" beside a phase that has not
    # started says exactly what it is, and a rule that flagged it would train
    # people to delete the honesty rather than the claim.
    invented = [(name, context) for name, context in _gate_check_claims(text)
                if name not in defined
                and not any(marker in context.lower() for marker in HONEST_MARKERS)]
    assert not invented, (
        f"{document.name}: presents a non-existent gate check as one — "
        + "; ".join(f"`{name}` in: {line[:110]}" for name, line in invented))


def test_the_rule_rejects_a_name_the_gate_does_not_define():
    """The counter-check for the inversion, and it names no inventions: a
    freely chosen name must be caught because the gate lacks it, not because
    a list says so."""
    defined = _check_names()
    invented = "cortex_map_consolidation"
    assert invented not in defined, "pick a name the gate really lacks"

    table = "\n".join([
        "| Phase | Inhalt | Gate-Check |",
        "|---|---|---|",
        f"| P9 | Kortikale Karte | `{invented}` |",
        "| P10 | Schichtregel | `layering` |",
    ])
    found = {name for name, _ in _gate_check_claims(table)}
    assert invented in found, "a new invention must be caught by the rule itself"
    assert "layering" in found and "layering" in defined


def test_a_name_outside_the_gate_check_column_is_not_a_claim():
    """The rule has to stay narrow in the other direction: the phase tables
    also carry module names, parameters and metrics, and flagging those would
    make the test noise that gets deleted rather than obeyed."""
    table = "\n".join([
        "| Phase | Inhalt | Gate-Check |",
        "|---|---|---|",
        "| P3 | `max_events_s` begrenzt `perception.py` | `layering` |",
    ])
    assert {name for name, _ in _gate_check_claims(table)} == {"layering"}


def test_the_documents_are_actually_being_read():
    assert len(DESIGN_DOCS) >= 5
    assert any(p.name == "DREAM-POD-DESIGN-20260920.md" for p in DESIGN_DOCS)
    assert _check_names(), "no gate checks were parsed at all"


def test_the_rule_catches_a_tick_on_a_red_check():
    """The counter-check, in the shape the defect had."""
    checks = _gate_checks()
    red = sorted(name for name, ok in checks.items() if not ok)
    assert red, "the gate is fully green; this counter-check needs a red check"

    line = f"| D3 | Reflex-Bindung | `{red[0]}` ✔ |"
    assert TICKED.findall(line) == [red[0]]
