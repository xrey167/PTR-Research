"""Deterministic semantic guard for the trained reader's tiny arithmetic task.

The model remains responsible for reading the Pod and producing an answer. The
guard is a lifecycle safety barrier: it recomputes the typed operation from
trusted evidence before an answer is committed, so a fluent but wrong Yes/No
cannot cross the boundary. Raw and guarded metrics are reported separately.
"""
from __future__ import annotations

import re


_NUMBERS = {"a": 1, "an": 1, "one": 1, "eine": 1, "ein": 1, "two": 2, "zwei": 2,
            "three": 3, "drei": 3, "four": 4, "vier": 4, "five": 5, "fünf": 5,
            "six": 6, "sechs": 6, "seven": 7, "sieben": 7, "eight": 8, "acht": 8,
            "nine": 9, "neun": 9}


def _number(value: str) -> int | None:
    return int(value) if value.isdigit() else _NUMBERS.get(value.casefold())


def _days(question: str) -> int | None:
    q = question.casefold()
    weeks = re.search(r"(\d+|one|eine|two|zwei|three|drei|four|vier|five|fünf|six|sechs|seven|sieben|eight|acht|nine|neun)\s+weeks?|"
                      r"(\d+|eine|ein|zwei|drei|vier|fünf|sechs|sieben|acht|neun)\s+wochen?", q)
    days = re.findall(r"(\d+)\s+days?|(?:(\d+|ein|eine|zwei|drei)\s+tage?)", q)
    total = 0
    if weeks:
        token = next(x for x in weeks.groups() if x is not None)
        total += (_number(token) or 0) * 7
    for pair in days:
        token = next(x for x in pair if x)
        total += _number(token) or 0
    return total or None


def guarded_answer(row: dict, raw: str) -> str:
    """Return the canonical answer only for recognized, typed reader tasks."""
    evidence = row.get("evidence")
    if not evidence:
        return "UNKNOWN"
    if row.get("task") == "multi_hop_total" and evidence.get("facts"):
        numbers = [int(x) for fact in evidence["facts"] for x in re.findall(r"\b\d+\b", fact)]
        if len(numbers) >= 2:
            total = sum(numbers)
            return f"{total} {'Tage' if row.get('language') == 'de' else 'days'}"
    value = int(evidence["lead_time_days"])
    question = row.get("question", "")
    q = question.casefold()
    language = row.get("language", "en")
    if "buffer" in q or "puffer" in q or "buffer" in row.get("task", "") or "puffer" in row.get("task", ""):
        extra = 4
        match = re.search(r"(\d+|one|eine|two|zwei|three|drei|four|vier)\s+"
                          r"(?:extra\s+)?(?:buffer|puffer)", q)
        if match:
            extra = _number(match.group(1)) or extra
        total = value + extra
        return f"{total} {'Tage' if language == 'de' else 'days'}"
    if ("deadline" in q or "frist" in q or "within" in q or "spätestens" in q
            or "spatestens" in q or "arrive" in q or "eintreffen" in q):
        deadline = _days(question)
        if deadline is not None:
            yes = deadline >= value
            return f"{'Ja' if yes and language == 'de' else 'Yes' if yes else 'Nein' if language == 'de' else 'No'}; {value} {'Tage' if language == 'de' else 'days'}"
    if "lookup" in row.get("task", "") or "lead" in q or "lieferzeit" in q:
        return f"{value} {'Tage' if language == 'de' else 'days'}"
    return raw
