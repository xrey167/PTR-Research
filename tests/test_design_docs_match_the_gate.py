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


@pytest.mark.parametrize("document", DESIGN_DOCS, ids=lambda p: p.name)
def test_no_document_names_a_gate_check_that_does_not_exist(document):
    """The other half. A phase table listing `improve_cycle` as its gate
    check, when no such check is defined, reads as a verified phase and is
    not one. Naming it as missing is fine; presenting it as a check is not."""
    defined = _check_names()
    text = document.read_text(encoding="utf-8-sig")

    invented = []
    for line in text.splitlines():
        # Only lines that present a name AS the gate check for something.
        if "Gate-Check" not in line and "gate check" not in line.lower() \
                and not line.lstrip().startswith("|"):
            continue
        # A name presented WITH an honest marker is not a claim. The markers
        # are explicit and few on purpose: "(offen)" beside a phase that has
        # not started says exactly what it is, and a rule that flagged it
        # would train people to delete the honesty rather than the claim.
        if any(marker in line.lower() for marker in HONEST_MARKERS):
            continue
        for name in re.findall(r"`([a-z_][a-z0-9_]{3,})`", line):
            if name.endswith((".py", ".md", ".json")) or "/" in name:
                continue
            if name in defined:
                continue
            # Only flag names that look like a check the gate would define.
            if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)+", name) and name in KNOWN_INVENTED:
                invented.append((name, line.strip()[:110]))
    assert not invented, (
        f"{document.name}: presents a non-existent gate check as one — "
        + "; ".join(f"`{name}` in: {line}" for name, line in invented))


#: Markers that turn a check name from a claim into a statement of fact.
HONEST_MARKERS = (
    "existiert nicht", "does not exist", "fehlt", "(offen)", "offen)",
    "nicht umgesetzt", "noch nicht", "rot", "geplant",
)

#: Names that were presented as gate checks and never existed. The set only
#: shrinks: a phase that grows a real check drops out of it, and a document
#: that starts presenting one of these as a check again goes red.
KNOWN_INVENTED = {
    "perception_stream", "improve_cycle", "latent_addressing", "dream_deep",
}


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
