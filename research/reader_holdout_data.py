"""A fresh, uncontaminated evaluation split for the reader.

Why this exists: `raw` exact match on the frozen test split has stood at
125/132 (94.7%) since Gen-6 and the concept family at 44/44, while `guarded`
has stood at 92/132 since Gen-3. A metric that saturated cannot tell Gen-8
from Gen-7, and a curriculum simulator fitted on it cannot predict anything.

This module builds a `holdout` split with the SAME case families and the same
row schema as `reader_training_data.build_data()`, so the numbers stay
comparable, but with entities, component codes, lead-time values, deadline
offsets and phrasings that appear in none of the frozen splits. The frozen
train/dev/test splits are not touched: `reader_training_data.py` is imported
read-only and its SPLITS table is left exactly as it is.

Disjointness is not a claim here, it is checked — by
`disjointness_report()` below, by tests/test_reader_holdout_data.py, and by
the `holdout_split_disjoint` gate check.
"""
from __future__ import annotations

from neural_pods.pod_types import pod_type_for_task
from research.reader_training_data import build_data, duration

SPLIT = "holdout"

# Disjoint from every name, code and value in reader_training_data.SPLITS.
SUPPLIERS = ["Kirano Fabrikation", "Lumeth Zulieferer"]
COMPONENTS = ["K93", "L04"]
LEAD_TIMES = [61, 66]
CUSTOMS_DAYS = {"K93": 9, "L04": 11}
BUFFER_DAYS = 4          # same offset as the frozen splits: keeps `buffer` comparable
DEADLINE_OFFSETS = [("short", -2), ("equal", 0), ("long", 3)]

# reader_training_data.duration() spells weeks with a ten-word vocabulary, so
# every deadline it has to render must stay under ten whole weeks. Checked
# here rather than discovered as an IndexError halfway through a run.
MAX_RENDERABLE_DAYS = 69
assert max(LEAD_TIMES) + max(offset for _name, offset in DEADLINE_OFFSETS) \
    <= MAX_RENDERABLE_DAYS, "deadline exceeds the week vocabulary of duration()"

# Surface forms that occur in no frozen split.
STEMS = {
    "en": "A framework agreement covers {part} delivered by {name}. ",
    "de": "Ein Rahmenvertrag umfasst {part} geliefert von {name}. ",
}
PROMPTS = {
    "en": ("report the procurement lead time in days.",
           "report the lead time after adding a four-day contingency.",
           "is the shipment due within {deadline} of the order date?"),
    "de": ("Gib die Beschaffungsfrist in Tagen an.",
           "Gib die Frist inklusive eines viertaegigen Zuschlags an.",
           "Ist die Sendung innerhalb von {deadline} ab Bestelldatum faellig?"),
}
FOLLOWUPS = {
    "en": ("Repeat the procurement lead time for that agreement.",
           "Raise that figure by four days and report it.",
           "Which lead time is in force for that agreement today?"),
    "de": ("Wiederhole die Beschaffungsfrist dieses Vertrags.",
           "Erhoehe diesen Wert um vier Tage und nenne ihn.",
           "Welche Frist gilt fuer diesen Vertrag heute?"),
}


def _target(answer, value: int, language: str) -> str:
    unit = "days" if language == "en" else "Tage"
    if answer is None:
        return "UNKNOWN"
    if type(answer) is bool:
        yes_no = ("Yes" if answer else "No") if language == "en" else ("Ja" if answer else "Nein")
        return f"{yes_no}; {value} {unit}"
    return f"{answer} {unit}"


def _row(case_id, task, language, evidence, question, history, target, assessment):
    return {"id": case_id, "task": task,
            "pod_type": pod_type_for_task(task).value, "language": language,
            "evidence": evidence, "question": question, "history": history,
            "target": target, "assessment": assessment}


def build_holdout() -> list[dict]:
    """Every case of the frozen families, on unseen entities and phrasings."""
    rows: list[dict] = []
    for name, part in zip(SUPPLIERS, COMPONENTS):
        for value in LEAD_TIMES:
            for language in ("en", "de"):
                stem = STEMS[language].format(part=part, name=name)
                lookup, buffer_prompt, deadline_prompt = PROMPTS[language]
                evidence = {"supplier": name, "component": part, "lead_time_days": value}

                cases = [("lookup", lookup, value, None),
                         ("buffer", buffer_prompt, value + BUFFER_DAYS, None)]
                for unit in ("days", "weeks"):
                    for relation, offset in DEADLINE_OFFSETS:
                        limit = value + offset
                        cases.append((f"deadline_{unit}_{relation}",
                                      deadline_prompt.format(
                                          deadline=duration(limit, language, unit)),
                                      value <= limit, limit))
                if value == LEAD_TIMES[0]:
                    # Missing evidence carries no value; one case per entity.
                    cases.append(("missing", lookup, None, None))

                for task, question, answer, limit in cases:
                    rows.append(_row(
                        f"{SPLIT}:{part}:{value}:{language}:{task}", task, language,
                        None if task == "missing" else dict(evidence),
                        stem + question, [], _target(answer, value, language),
                        {"supplier": name, "component": part, "value": value,
                         "deadline_days": limit, "answer": answer}))

                other = LEAD_TIMES[(LEAD_TIMES.index(value) + 1) % len(LEAD_TIMES)]
                rows.extend(_dialogue_rows(evidence, language, stem, other))
    rows.extend(_multi_hop_rows())
    return rows


def _dialogue_rows(evidence, language, context, prior_value) -> list[dict]:
    value = evidence["lead_time_days"]
    unit = "days" if language == "en" else "Tage"
    rows = []
    for task, question, answer in zip(
            ("followup_lookup", "followup_buffer", "stale_followup"),
            FOLLOWUPS[language], (value, value + BUFFER_DAYS, value)):
        history = [{"role": "user", "content": context.strip()}]
        if task == "stale_followup":
            history.append({"role": "assistant", "content": (
                f"Superseded figure: {prior_value} days." if language == "en"
                else f"Ueberholter Wert: {prior_value} Tage.")})
        rows.append(_row(
            f"{SPLIT}:{evidence['component']}:{value}:{language}:{task}", task,
            language, dict(evidence), question, history,
            f"{answer} {unit}",
            {"supplier": evidence["supplier"], "component": evidence["component"],
             "value": value, "answer": answer, "deadline_days": None,
             "prior_value": prior_value if task == "stale_followup" else None}))
    return rows


def _multi_hop_rows() -> list[dict]:
    rows = []
    for name, part in zip(SUPPLIERS, COMPONENTS):
        lead, customs = LEAD_TIMES[0], CUSTOMS_DAYS[part]
        total = lead + customs
        for language in ("en", "de"):
            if language == "en":
                question = (f"Across the framework agreement, {part} from {name} "
                            f"ships in {lead} days and clearance adds {customs} "
                            "days; what is the combined lead time?")
                target, unit = f"{total} days", "days"
                facts = [f"Framework agreement: {name} ships {part} in {lead} days.",
                         f"Clearance for {part} adds {customs} days."]
            else:
                question = (f"Im Rahmenvertrag versendet {name} die Komponente "
                            f"{part} in {lead} Tagen, die Abfertigung dauert "
                            f"{customs} Tage; wie lang ist die Gesamtfrist?")
                target, unit = f"{total} Tage", "Tage"
                facts = [f"Rahmenvertrag: {name} versendet {part} in {lead} Tagen.",
                         f"Die Abfertigung von {part} dauert {customs} Tage."]
            rows.append(_row(
                f"{SPLIT}:multihop:{part}:{language}", "multi_hop_total", language,
                {"facts": facts}, question, [], target,
                {"supplier": name, "component": part, "value": total,
                 "answer": total, "deadline_days": None, "hop_count": 2}))
    return rows


def disjointness_report(frozen: dict[str, list[dict]] | None = None) -> dict:
    """What the holdout split shares with the frozen splits. Empty is the goal.

    Compared: case ids, full question strings, targets, and the supplier,
    component and lead-time values recorded in each row's assessment.
    """
    frozen = frozen if frozen is not None else build_data()
    holdout = build_holdout()
    frozen_rows = [row for rows in frozen.values() for row in rows]

    def _facet(rows, key):
        return {row["assessment"].get(key) for row in rows
                if row.get("assessment") and row["assessment"].get(key) is not None}

    overlaps = {
        "ids": sorted({r["id"] for r in holdout} & {r["id"] for r in frozen_rows}),
        "questions": sorted({r["question"] for r in holdout}
                            & {r["question"] for r in frozen_rows}),
        "suppliers": sorted(_facet(holdout, "supplier") & _facet(frozen_rows, "supplier")),
        "components": sorted(_facet(holdout, "component") & _facet(frozen_rows, "component")),
        "values": sorted(_facet(holdout, "value") & _facet(frozen_rows, "value")),
    }
    return {
        "holdout_rows": len(holdout),
        "frozen_rows": len(frozen_rows),
        "families": sorted({r["task"] for r in holdout}),
        "frozen_families": sorted({r["task"] for r in frozen_rows}),
        "overlaps": overlaps,
        "disjoint": all(not v for v in overlaps.values()),
    }
