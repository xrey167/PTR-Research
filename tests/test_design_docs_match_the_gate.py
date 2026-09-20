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
from typing import NamedTuple

import pytest

import verify_architecture_gate as gate

REPO = Path(__file__).resolve().parents[1]
DESIGN_DOCS = sorted(
    list((REPO / "research").glob("*DESIGN*.md"))
    + [REPO / "ARCHITECTURE-MASTER-20260920.md"])

#: `check_name` followed by a tick, in a table cell or prose. Both ticks:
#: the documents use U+2714 and U+2713 interchangeably, and matching only
#: the first one made a whole table invisible to this rule.
TICKED = re.compile(r"`([a-z_][a-z0-9_]*)`\s*(?:\*\*)?[✔✓]")


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

#: The same, unbackticked. Only read inside the gate-check COLUMN, which is
#: by definition the column of check names: ARCHITECTURE-MASTER's component
#: table writes them plain ("tests, authenticated_transport"), and the whole
#: table was invisible to this file until it did.
BARE_NAME = re.compile(r"(?<![`\w])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![`\w])")

#: Prose that presents what follows as the gate check for something. The
#: anchor is the words themselves, so a name only counts when the document
#: called it a gate check.
GATE_CHECK = re.compile(r"[Gg]ate[-‑ ]?[Cc]hecks?\b")

#: How far past the anchor a name still belongs to it, and how far before
#: the anchor a marker still qualifies the first name after it.
PROSE_WINDOW = 120
PROSE_LEAD = 60

#: Markers that turn a check name from a claim into a statement of fact.
HONEST_MARKERS = (
    "existiert nicht", "does not exist", "fehlt", "(offen)", "offen)",
    "nicht umgesetzt", "noch nicht", "rot", "geplant",
)


class Claim(NamedTuple):
    """One name a document presents as a gate check.

    `scope` is the text that qualifies THIS name — from the name to the next
    one — and `display` is the whole line, for the failure message. They were
    the same string once, which meant one marked name exempted every other
    name beside it: `` `household_join` (geplant) · `cortex_map` `` passed.
    """
    name: str
    scope: str
    display: str


def _is_separator(line: str) -> bool:
    """`|---|---|`, the row that marks the line above it as a header."""
    return (line.startswith("|") and "-" in line
            and set(line) <= set("|-: "))


def _names_with_scope(segment: str, display: str) -> list[Claim]:
    """Split `segment` at its names; each name owns the text up to the next.

    Every honest marker in these documents follows the name it qualifies —
    `` `dream_deep` (offen) ``, `` `latent_addressing` **existiert nicht** ``,
    `` `reflex_failover` **grün** · `reflex_dispatch` **rot bis P5** `` — so
    that is the span this reads, rather than the whole row.
    """
    found = sorted(
        [(match.start(), match.end(), match.group(1))
         for match in NAME.finditer(segment)]
        + [(match.start(), match.end(), match.group(1))
           for match in BARE_NAME.finditer(segment)])
    if not found:
        return []
    # Text BEFORE the first name qualifies every name in the segment: it
    # cannot belong to one of them, and "geplant: `a`, `b`, `c`" is how a
    # roadmap row says that none of the three exists yet. Text after a name
    # belongs to that name alone — which is the whole point of this split.
    prefix = segment[:found[0][0]]
    claims = []
    for index, (_, name_end, name) in enumerate(found):
        scope_end = found[index + 1][0] if index + 1 < len(found) else len(segment)
        claims.append(Claim(name, prefix + segment[name_end:scope_end], display))
    return claims


def _gate_check_claims(text: str) -> list[Claim]:
    """Every name a document presents AS a gate check, with its own scope.

    Two places do that, and both are read here rather than guessed at:

      * the gate-check column of a phase table, found by its own header, and
      * prose that says "Gate-Check" and then names one.

    A name anywhere else — a module, a parameter, a metric in a neighbouring
    column — is not a claim about the gate and is not collected. That
    distinction is the whole rule. It used to be drawn by a hand-kept set of
    names already known to be invented, which let every NEW invention
    straight through.
    """
    claims: list[Claim] = []

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
            claims.extend(_names_with_scope(cells[column], stripped))

    # Prose. Newlines become spaces so a claim wrapped across two lines is
    # still one claim; a cell or sentence boundary ends the window, because
    # the next cell is about something else. Backticks are required here:
    # outside the gate-check column, a bare identifier is usually a module.
    #
    # The scope is asymmetric to the table's on purpose. In a gate-check cell
    # the marker always follows its name, but a sentence puts it wherever it
    # reads best — "D3 gebaut, Gate-Check rot: ... `dream_reflex`" marks the
    # name from in front of the anchor. So the FIRST name after an anchor
    # also owns the run-up to it; every later name starts where the previous
    # one ended, which is what keeps one marked name from covering the next.
    flat = text.replace("\n", " ")
    for match in GATE_CHECK.finditer(flat):
        lead = flat[max(0, match.start() - PROSE_LEAD):match.start()]
        window = re.split(r"\||\.\s", flat[match.end():match.end() + PROSE_WINDOW])[0]
        display = flat[max(0, match.start() - 40):match.end() + 80].strip()
        found = list(NAME.finditer(window))
        for index, name in enumerate(found):
            scope_start = found[index - 1].end() if index else 0
            scope_end = found[index + 1].start() if index + 1 < len(found) else len(window)
            scope = window[scope_start:name.start()] + window[name.end():scope_end]
            claims.append(Claim(name.group(1),
                                (lead + scope) if index == 0 else scope,
                                display))
    return claims


def _marked_honestly(claim: Claim) -> bool:
    """Whether this name carries a marker that turns it into a statement.

    The markers are explicit and few on purpose: "(offen)" beside a phase
    that has not started says exactly what it is, and a rule that flagged it
    would train people to delete the honesty rather than the claim.
    """
    return any(marker in claim.scope.lower() for marker in HONEST_MARKERS)


def _invented_claims(text: str, defined: set[str]) -> list[Claim]:
    """Names presented as gate checks that the gate does not define.

    The production test and its counter-check both go through here. They used
    not to: the counter-check called `_gate_check_claims` and stopped at
    "the name was extracted", so the filter it was meant to protect was never
    executed by it — which is why the row-wide-scope defect above survived a
    counter-check written in the same commit.
    """
    return [claim for claim in _gate_check_claims(text)
            if claim.name not in defined and not _marked_honestly(claim)]


#: The one check whose verdict is a statement about the WORKING TREE rather
#: than about recorded evidence: `tests` compares `sources_sha256` against
#: the tree as it is right now, so it is red in every checkout with an
#: unrecorded edit — including, routinely, the one this test runs in. A
#: document cannot be written to track that, so the red-check rule skips it.
#: Nothing else is skipped; this is a property of the check, not a name that
#: happened to be inconvenient.
TREE_DEPENDENT = frozenset({"tests"})


def _unmarked_red_claims(text: str, checks: dict[str, bool]) -> list[Claim]:
    """Names the gate defines and refuses, presented without saying so."""
    return [claim for claim in _gate_check_claims(text)
            if claim.name not in TREE_DEPENDENT
            and checks.get(claim.name) is False
            and not _marked_honestly(claim)]


@pytest.mark.parametrize("document", DESIGN_DOCS, ids=lambda p: p.name)
def test_no_document_names_a_gate_check_that_does_not_exist(document):
    """The other half. A phase table listing `improve_cycle` as its gate
    check, when no such check is defined, reads as a verified phase and is
    not one. Naming it as missing is fine; presenting it as a check is not.

    This rule is the inverse of what it was. It used to flag only names on a
    hand-kept list of inventions already found — so a name invented tomorrow
    passed, and the test written to close that class of mistake reproduced
    it. Every name a document presents as a gate check must be one the gate
    defines.
    """
    invented = _invented_claims(
        document.read_text(encoding="utf-8-sig"), _check_names())
    assert not invented, (
        f"{document.name}: presents a non-existent gate check as one — "
        + "; ".join(f"`{c.name}` in: {c.display[:110]}" for c in invented))


@pytest.mark.parametrize("document", DESIGN_DOCS, ids=lambda p: p.name)
def test_no_document_presents_a_red_check_without_saying_it_is_red(document):
    """The third face of the same mistake. The first two rules ask whether a
    name is ticked and whether it exists. Neither sees a table headed
    "Implementiert + validiert (Gate-Checks)" that lists `dream_pipeline`,
    `mesh_cache` and `reflex_dispatch` — all three red — with no tick and no
    marker: the heading does the claiming.

    A refused check named as the gate check for something has to say so.
    `rot` is in the marker list, so one word in the cell is the whole cost.
    """
    unmarked = _unmarked_red_claims(
        document.read_text(encoding="utf-8-sig"), _gate_checks())
    assert not unmarked, (
        f"{document.name}: names a RED gate check without marking it — "
        + "; ".join(f"`{c.name}` in: {c.display[:110]}" for c in unmarked))


def test_the_rule_rejects_a_name_the_gate_does_not_define():
    """The counter-check for the inversion, through the SAME filter the
    production test uses — and it names no inventions: a freely chosen name
    is caught because the gate lacks it, not because a list says so.

    The row holds both cases at once, which is the case that was fail-open:
    a marked undefined name beside an unmarked one.
    """
    defined = _check_names()
    invented = "cortex_map_consolidation"
    planned = "cortex_map_planned"
    assert {invented, planned}.isdisjoint(defined), "pick names the gate lacks"

    table = "\n".join([
        "| Phase | Inhalt | Gate-Check |",
        "|---|---|---|",
        f"| P9 | Kortikale Karte | `{planned}` (geplant) · `{invented}` |",
        "| P10 | Schichtregel | `layering` |",
    ])
    reported = {claim.name for claim in _invented_claims(table, defined)}
    assert invented in reported, "a new invention must be caught by the rule"
    assert planned not in reported, "a marked name is a statement, not a claim"
    assert "layering" not in reported and "layering" in defined


def test_the_red_check_rule_needs_the_marker_beside_that_name():
    """Counter-check for the third rule, in both directions and with the same
    trap: a marked red check beside an unmarked one."""
    checks = _gate_checks()
    red = sorted(name for name, ok in checks.items() if not ok)
    green = sorted(name for name, ok in checks.items() if ok)
    assert red and green, "this counter-check needs one of each"

    table = "\n".join([
        "| Komponente | Datei | Gate-Check |",
        "|---|---|---|",
        f"| A | a.py | `{red[0]}` **rot** · `{red[1]}` |",
        f"| B | b.py | `{green[0]}` |",
    ])
    reported = {claim.name for claim in _unmarked_red_claims(table, checks)}
    assert reported == {red[1]}, (
        "the marker belongs to the name it follows, and a green check is "
        f"never reported; got {sorted(reported)}")


def test_a_marker_before_the_first_name_covers_the_whole_cell():
    """A roadmap row names several checks that do not exist yet, and writing
    "(geplant)" three times reads worse than saying it once in front. Text
    before the first name cannot belong to one of them, so it qualifies all
    of them — and text AFTER a name still belongs to that name alone."""
    defined = _check_names()
    table = "\n".join([
        "| Phase | Gate-Checks (neu) | Endstand |",
        "|---|---|---|",
        "| S2–S5 | geplant: `raft_jemalloc`, `work_stealing` | 49/49 |",
        "| X | `raft_jemalloc`, `work_stealing` | — |",
    ])
    reported = [c for c in _invented_claims(table, defined)]
    assert {c.name for c in reported} == {"raft_jemalloc", "work_stealing"}
    assert all("| X |" in c.display for c in reported), (
        "only the unmarked row may be reported")


def test_a_name_outside_the_gate_check_column_is_not_a_claim():
    """The rule has to stay narrow in the other direction: the phase tables
    also carry module names, parameters and metrics, and flagging those would
    make the test noise that gets deleted rather than obeyed."""
    table = "\n".join([
        "| Phase | Inhalt | Gate-Check |",
        "|---|---|---|",
        "| P3 | `max_events_s` begrenzt `perception.py` | `layering` |",
    ])
    assert {claim.name for claim in _gate_check_claims(table)} == {"layering"}


def test_an_unbackticked_name_in_the_gate_check_column_is_a_claim():
    """ARCHITECTURE-MASTER's component table writes its check names plain.
    Requiring backticks made that whole table invisible to both rules, which
    is how it came to list three red checks under "validiert"."""
    table = "\n".join([
        "| Komponente | Datei | Gate-Check(s) |",
        "|---|---|---|",
        "| Registry | registry.py | tests, authenticated_transport |",
    ])
    assert {claim.name for claim in _gate_check_claims(table)} == {
        "authenticated_transport"}, "bare names need an underscore to qualify"


def test_the_skipped_check_is_the_one_that_reads_the_working_tree():
    """TREE_DEPENDENT has one entry, and this is why it may: the `tests`
    check compares a recorded `sources_sha256` against the tree as it is
    right now. A second name could only get in here by also being read out
    of the source tree — and this test would have to be widened on purpose."""
    source = (REPO / "research" / "verify_architecture_gate.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    verify = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "verify")
    checks = next(node.value for node in ast.walk(verify)
                  if isinstance(node, ast.Assign)
                  and any(getattr(t, "id", "") == "checks" for t in node.targets))
    reads_the_tree = {
        key.value for key, value in zip(checks.keys, checks.values)
        if any(isinstance(call.func, ast.Name)
               and call.func.id == "source_fingerprint"
               for call in ast.walk(value) if isinstance(call, ast.Call))}
    assert reads_the_tree == set(TREE_DEPENDENT), (
        f"the gate reads the source tree in {sorted(reads_the_tree)}, "
        f"the skip list says {sorted(TREE_DEPENDENT)}")


def test_the_tick_rule_sees_both_ticks():
    """U+2714 and U+2713 look alike and read alike. Matching one of them is
    how `mesh_cache`'s row stayed unnoticed."""
    for tick in ("\u2714", "\u2713"):
        assert TICKED.findall(f"| D3 | Bindung | `dream_reflex` {tick} |") == [
            "dream_reflex"], tick


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
