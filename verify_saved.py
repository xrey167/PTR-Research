"""Fresh-process verification on new questions, with explicitly grounded RAG.

No training or selection on these new held-out templates. Persists every output.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import time
from neural_pods.data import FACTS
from neural_pods.model import PodModel
from neural_pods.registry import Registry, InvalidState, verify_files, hash_files
from neural_pods.routing import Router

TEMPLATES = [
    "For {c} sourced from {e}, specify the delivery lead time as a count of days.",
    "I am scheduling a {c} shipment with {e}. What is their recorded delivery duration?",
    "According to the supplier record, how many days are needed to receive {c} from {e}?",
    "Nenne die gespeicherte Lieferdauer in Tagen fuer Bauteil {c} beim Lieferanten {e}.",
]


def numeric_correct(text, value):
    return re.findall(r"\d+(?:\.\d+)?", text) == [str(value)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    original = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if original["status"] != "completed": raise RuntimeError("Wait for completed training")
    result = {"protocol": "new questions after initial pilot; fixed grounded-RAG instruction; no further training",
              "records": [], "summary": {}, "base_model_unchanged": False}
    registry = Registry(run / "registry.sqlite3")
    llm = PodModel(original["models"]["qwen"]["path"])
    fingerprint = llm.frozen_hash()
    if fingerprint != original["base_weights_sha256_before"]: raise RuntimeError("Base model changed")
    result["base_model_unchanged"] = True
    router = Router.load(original["models"]["encoder"]["path"], registry, run)
    for row in original["training"]:
        name = row["name"]
        if name not in {"all_facts", "pod0_g1", "pod1_g1", "pod2_g1"}: continue
        # Compare physical files with their original registered immutable artifacts.
        nodes = registry.db.execute("SELECT payload FROM nodes WHERE kind='lora'").fetchall()
        matches = [json.loads(n[0])["payload"] for n in nodes if json.loads(n[0])["payload"].get("adapter") == name]
        expected = matches[0]["files"]
        verify_files(row["path"], expected)
        llm.model.load_adapter(row["path"], adapter_name=name, is_trainable=False)

    for mode in ["base", "grounded_rag", "always_loaded", "routed_pod"]:
        for fact in FACTS:
            for template in TEMPLATES:
                question = template.format(e=fact.entity, c=fact.component)
                started = time.perf_counter()
                record = {"mode": mode, "question": question, "key": fact.key, "expected": fact.answer}
                try:
                    snapshot = None
                    adapter = "all_facts" if mode == "always_loaded" else None
                    evidence = None
                    if mode in {"grounded_rag", "routed_pod"}:
                        selected = router.select(question, learned=mode == "routed_pod")
                        dependencies = [selected["vector_artifact"]]
                        if mode == "routed_pod":
                            adapter = selected["adapter"]
                            dependencies += [selected["pod_artifact"], router.artifact]
                        else:
                            evidence = selected["evidence"]
                        snapshot = registry.snapshot(dependencies)
                        record["selected_key"] = selected["key"]
                    record.update(llm.generate(question, adapter=adapter, evidence=evidence))
                    if snapshot:
                        record["receipt"] = registry.commit(snapshot, record["text"])["answer_id"]
                except InvalidState as exc:
                    record.update(text="", blocked=str(exc))
                record["total_s"] = time.perf_counter() - started
                record["exact"] = record["text"].strip().lower().rstrip(".! ") == fact.answer
                record["numeric_correct"] = numeric_correct(record["text"], fact.value)
                result["records"].append(record)
        rows = [r for r in result["records"] if r["mode"] == mode]
        result["summary"][mode] = {"exact": sum(r["exact"] for r in rows),
              "numeric_correct": sum(r["numeric_correct"] for r in rows), "n": len(rows),
              "mean_latency_s": statistics.mean(r["total_s"] for r in rows),
              "mean_input_tokens": statistics.mean(r.get("input_tokens", 0) for r in rows)}
        print(json.dumps({"mode": mode, **result["summary"][mode]}), flush=True)
        (run / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    # Actual final runtime rejects previously revoked weights and permits restored state.
    revoked = registry.db.execute("SELECT id FROM nodes WHERE kind='lora' AND revoked=1").fetchall()
    result["revoked_artifacts_rejected"] = []
    for row in revoked:
        try:
            registry.snapshot([row[0]])
            result["revoked_artifacts_rejected"].append(False)
        except InvalidState:
            result["revoked_artifacts_rejected"].append(True)
    result["source_files_sha256"] = {"neural_pods/" + k: v for k, v in hash_files(Path("neural_pods")).items() if not k.startswith("__pycache__/")}
    result["status"] = "completed"
    (run / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    router.close()
    registry.close()


if __name__ == "__main__": main()
