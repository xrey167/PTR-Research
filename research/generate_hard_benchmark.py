"""Generate a leakage-controlled retrieval benchmark pilot."""
from __future__ import annotations
import json
from pathlib import Path


def build():
    suppliers = [(f"Supplier {i:02d}", f"VND-{chr(65+i//26)}{i:02d}", 10 + (i * 7) % 37) for i in range(30)]
    components = [f"component-{x}" for x in ("x12", "q7", "m4", "z9", "r2")]
    corpus, queries = [], []
    for i, (name, alias, lead) in enumerate(suppliers):
        split = "train" if i < 20 else "test"
        for j, component in enumerate(components):
            key = f"supplier:{i}:{component}:lead_time"
            corpus.append({"doc_id": f"doc-{i}-{j}", "knowledge_key": key,
                           "origin_key": f"src:erp:root:{i:02d}", "supplier": name,
                           "text": f"Procurement ledger record {alias} for {component}: transit window {lead} calendar days.",
                           "split": split})
            if split == "test":
                # Query aliases are deliberately held out from the corpus
                # surface form; a resolver must use provenance/entity links.
                held_out_alias = f"partner-{i:02d}-alias"
                phrasing = [
                    f"What transit window is recorded for {held_out_alias} on {component}?",
                    f"Using the internal partner handle {held_out_alias}, give the delivery duration for {component}.",
                    f"How long should procurement budget for {component} from {held_out_alias}?",
                ][j % 3]
                queries.append({"id": f"hard-{i}-{j}", "question": phrasing,
                                "answer": f"{lead} calendar days", "target_doc": f"doc-{i}-{j}",
                                "knowledge_key": key, "origin_key": f"src:erp:root:{i:02d}",
                                "distractor_docs": [f"doc-{k}-{j}" for k in range(20, 30) if k != i][:3]})
    return {"corpus": corpus, "queries": queries,
            "splits": {"corpus_train_entities": 20, "query_test_entities": 10},
            "scope": "leakage-controlled pilot; synthetic records"}


if __name__ == "__main__":
    report = build()
    Path("runs/hard-benchmark-pilot-001.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"corpus": len(report["corpus"]), "queries": len(report["queries"]), "splits": report["splits"]}, indent=2))
