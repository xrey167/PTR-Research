"""Generate a leakage-controlled multi-hop retrieval benchmark.

The test split holds out both provenance roots and question templates.  Each
question requires three source records: supplier transit, customs delay and
warehouse handling.  Distractors share entity and procurement vocabulary.
"""
from __future__ import annotations
import json
from pathlib import Path

TEMPLATES = [
    "For {alias}, what total elapsed time should the planner use for {part}?",
    "Give the end-to-end window for {part} when the partner is known internally as {alias}.",
    "How many calendar days from dispatch to shelf for {alias} / {part}?",
    "Combine transport, border clearance, and receiving for {alias} on {part}.",
]

def build(n_entities: int = 40):
    docs, queries = [], []
    for i in range(n_entities):
        supplier = f"Vendor {i:02d}"
        alias = f"internal-handle-{i:02d}"
        part = f"assembly-{chr(65 + i % 26)}{i:02d}"
        transit, customs, handling = 8 + i % 9, 2 + (i * 3) % 6, 1 + i % 4
        root = f"src:erp:{i:02d}"
        split = "train" if i < n_entities // 2 else "test"
        facts = [
            ("transport", f"{supplier} dispatches {part}; standard transit is {transit} calendar days."),
            ("customs", f"{supplier} {part} import clearance normally takes {customs} calendar days."),
            ("warehouse", f"Receiving and put-away for {supplier} {part} takes {handling} calendar days."),
        ]
        for kind, text in facts:
            docs.append({"doc_id": f"doc-{i}-{kind}", "origin_key": root,
                         "entity": supplier, "part": part, "kind": kind,
                         "text": text, "split": split})
        # Vocabulary-sharing distractors come from other held-out roots.
        if split == "test":
            template = TEMPLATES[(i - n_entities // 2) % len(TEMPLATES)]
            # Both entity and component surface forms are held out.  A
            # semantic compiler would resolve these handles before retrieval.
            query_part = f"line-code-{i:02d}"
            q = template.format(alias=alias, part=query_part)
            target = [f"doc-{i}-{k}" for k, _ in facts]
            distractors = [d["doc_id"] for d in docs if d["split"] == "test"
                           and d["origin_key"] != root and d["kind"] in {"transport", "customs"}][-6:]
            queries.append({"id": f"gen-{i:02d}", "question": q, "alias": alias,
                            "part": part, "part_alias": query_part, "target_docs": target,
                            "origin_key": root, "answer_days": transit + customs + handling,
                            "distractor_docs": distractors})
    return {"corpus": docs, "queries": queries,
            "splits": {"train_roots": n_entities // 2, "test_roots": n_entities // 2,
                       "held_out_templates": True, "held_out_aliases": True},
            "scope": "synthetic leakage-controlled multi-hop retrieval benchmark"}

if __name__ == "__main__":
    out = Path("runs/generalization-benchmark-001.json")
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"corpus": 120, "queries": 20, "scope": build()["scope"]}, indent=2))
