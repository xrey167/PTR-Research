"""Compile the existing typed reader curriculum into a hard-tagged mix."""
from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reader_training_data import build_data
from neural_pods.pod_taxonomy import REQUIRED_TRAINING_TAGS, validate_training_record


def key(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def compile_mix():
    source = build_data()
    output = {}
    for source_split, rows in source.items():
        split = "validation" if source_split == "dev" else source_split
        compiled = []
        for row in rows:
            kind = row["pod_type"]
            evidence = json.dumps(row.get("evidence"), sort_keys=True, ensure_ascii=False)
            identity = f"{kind}|{row.get('task')}|{evidence}"
            tags = set(REQUIRED_TRAINING_TAGS[kind])
            tags.update({f"domain:procurement", f"lang:{row.get('language','en')}"})
            item = {"id": row["id"], "pod_type": kind, "domain": "procurement",
                    "semantic_role": row.get("task", kind), "tags": sorted(tags),
                    "origin_keys": [key("origin", evidence)],
                    "knowledge_key": key("knowledge", identity),
                    "generation_key": key("generation", identity + "|" + split),
                    "split": split, "input": row["question"], "target": row["target"],
                    "task": row.get("task"), "language": row.get("language"),
                    "evidence": row.get("evidence")}
            validate_training_record(item)
            compiled.append(item)
        output[split] = compiled
    return output


def main():
    out_dir = Path(__file__).resolve().parents[1] / "runs" / "pod-training-mix-001"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = compile_mix()
    (out_dir / "dataset.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"status": "compiled_not_trained", "rows": {k: len(v) for k, v in data.items()},
              "pod_types": Counter(r["pod_type"] for rows in data.values() for r in rows),
              "required_tags_validated": True,
              "next_step": "train reader LoRA with frozen test split and compare base/retrieval/adapter"}
    report["pod_types"] = dict(report["pod_types"])
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
