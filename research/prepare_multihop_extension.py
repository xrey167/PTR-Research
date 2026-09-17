"""Prepare a leakage-controlled three-hop reader curriculum.

The existing multihop adapter was trained on two-hop supplier+customs totals.
This extension adds a third, independent warehouse leg and holds out entities
and formulations.  It is deliberately only a preparation step: training is
run by ``train_reader.py`` on a GPU host.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


CASES = {
    "train": [("Avelin Components", "A31", 14, 3, 2), ("Branor Supply", "B42", 18, 5, 4),
               ("Cedrin Parts", "C53", 21, 2, 6), ("Dovira Works", "D64", 25, 4, 1)],
    "dev": [("Evrana Components", "E75", 35, 6, 3), ("Faldor Supply", "F86", 39, 1, 5)],
    "test": [("Gavren Parts", "G17", 52, 7, 8), ("Helvona Works", "H28", 56, 3, 9)],
}


def rows(split):
    result = []
    for supplier, component, transit, customs, warehouse in CASES[split]:
        total = transit + customs + warehouse
        for lang in ("en", "de"):
            if lang == "en":
                q = (f"Compute the final arrival time for {component} from {supplier}: "
                     f"supplier transport is {transit} days, customs takes {customs} days, "
                     f"and warehouse handling takes {warehouse} days.")
                target = f"{total} days"
                facts = [f"Supplier transport: {transit} days.", f"Customs: {customs} days.", f"Warehouse handling: {warehouse} days."]
            else:
                q = (f"Berechne die endgültige Ankunftszeit für {component} von {supplier}: "
                     f"Lieferant {transit} Tage, Zoll {customs} Tage und Lager {warehouse} Tage.")
                target = f"{total} Tage"
                facts = [f"Lieferantentransport: {transit} Tage.", f"Zoll: {customs} Tage.", f"Lagerbearbeitung: {warehouse} Tage."]
            result.append({"id": f"{split}:three-hop:{component}:{lang}", "task": "multi_hop_total",
                           "pod_type": "reasoning", "language": lang, "evidence": {"facts": facts},
                           "question": q, "history": [], "target": target,
                           "assessment": {"supplier": supplier, "component": component, "value": total,
                                          "answer": total, "hop_count": 3}})
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = {split: rows(split) for split in CASES}
    for split, values in data.items():
        (args.output / f"{split}.json").write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"schema": "three-hop-extension:v1", "rows": {k: len(v) for k, v in data.items()},
              "hop_count": 3, "held_out_entities": CASES["test"],
              "training_status": "prepared_not_trained", "scope": "synthetic curriculum; no quality claim"}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
