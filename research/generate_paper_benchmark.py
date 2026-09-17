"""Generate a deterministic provenance-labelled pilot benchmark skeleton."""
from __future__ import annotations

import json
from pathlib import Path


SUPPLIERS = [("Müller GmbH", "muller", 24), ("Kern AG", "kern", 18), ("Nova Parts", "nova", 31)]
TEMPLATES = [
    ("context", "Which supplier handles {component}?", "supplier"),
    ("math", "How many days does {supplier} need for {component}?", "lead_time"),
    ("model", "Which approval rule applies to {supplier} for {component}?", "approval"),
]


def build(count: int = 700):
    rows = []
    for i in range(count):
        supplier, alias, lead = SUPPLIERS[i % len(SUPPLIERS)]
        component = f"X{12 + (i % 7)}"
        kind, template, predicate = TEMPLATES[i % len(TEMPLATES)]
        question = template.format(supplier=supplier, component=component)
        if predicate == "supplier":
            answer = supplier
        elif predicate == "lead_time":
            answer = f"{lead} days"
        else:
            answer = "approval required above 25000 EUR"
        origin = f"src:sap:po:{10000 + i % 30}"
        rows.append({
            "id": f"q{i:04d}", "split": "train" if i < 300 else "validation" if i < 400 else "test",
            "question": question, "answer": answer, "pod_type": kind,
            "knowledge_key": f"supplier:{alias}:{component.lower()}:{predicate}",
            "generation_key": f"generation:{1 + i % 3}", "origin_keys": [origin],
            "independent_root_count": 1, "distractor_keys": [f"src:sap:po:{20000 + i % 11}"],
        })
    return rows


if __name__ == "__main__":
    rows = build()
    Path("runs").mkdir(exist_ok=True)
    path = Path("runs/paper-benchmark-pilot-001.jsonl")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"path": str(path), "rows": len(rows),
                      "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}}, indent=2))
