"""Real Qwen LoRA + typed semantic compiler + filtered ANN + lifecycle demo."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time
from sentence_transformers import SentenceTransformer
from neural_pods.model import PodModel
from neural_pods.registry import Registry, InvalidState, digest, hash_files, verify_files
from neural_pods.semantics import SemanticCompiler
from neural_pods.semantic_routing import SemanticRouter

TRAIN = [
    "What is the lead time for X12 from Müller GmbH?",
    "How many days does Müller GmbH need to deliver X12?",
    "State the delivery time of X12 from Müller GmbH.",
    "Give the lead time in days for X12, supplied by Müller GmbH.",
    "What delivery delay should I plan for Müller GmbH and X12?",
    "How long is the procurement lead time for Müller GmbH X12?",
    "Tell me the delivery duration in days of Müller GmbH for X12.",
    "What is the recorded lead time for Müller GmbH, component X12?",
]
HELD_OUT = [
    "Current delivery lead time for supplier Müller?",
    "X12 procurement delay from Müller GmbH?",
    "How long does Mueller need for X12?",
    "Wie lange braucht Muller GmbH fuer X12?",
    "Welche Lieferzeit hat Müller fuer X12?",
    "What is the delivery lead time for Muller?",
]


def hard_record(value, version):
    return {"subject_id": "supplier:muller", "subject_label": "Müller GmbH", "component_id": "x12",
            "component_label": "X12", "predicate": "lead_time", "object": {"value": value, "unit": "days"},
            "role": "FACT", "type": "supplier_metric", "owner": "demo-procurement", "acl": ["buyer"],
            "trusted_aliases": ["Müller", "Muller GmbH", "Mueller", "Mueller GmbH"],
            "trusted_tags": ["supplier", "lead-time"], "domain": "procurement", "language": "de",
            "valid_from": "2026-09-01T00:00:00+00:00", "valid_until": None,
            "source": {"system": "fixture:sap", "record_id": "PO_HISTORY_92831", "version": str(version),
                       "content": {"supplier_id": "muller", "part": "X12", "lead_time_days": value, "fictional": True}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("semantic-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    report = {"status": "running", "models": manifest, "fictional_source_data": True,
              "training_questions": TRAIN, "held_out_questions": HELD_OUT, "training": [], "answers": [], "lifecycle": {}}
    started = time.perf_counter()
    def save():
        report["elapsed_s"] = time.perf_counter() - started
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    registry = Registry(out / "registry.sqlite3")
    compiler = SemanticCompiler(registry)
    llm = PodModel(manifest["qwen"]["path"])
    base_hash = llm.frozen_hash()
    report["base_weights_sha256"] = base_hash
    encoder = SentenceTransformer(manifest["encoder"]["path"], device="cpu", local_files_only=True)
    router = SemanticRouter(registry, encoder, out)
    handles = {}

    def build(value, version):
        cko = compiler.compile(hard_record(value, version),
                [{"field": "topics", "value": ["supply-chain"], "source": "demo-enricher", "confidence": 0.8}], principal="buyer")
        adapter = f"muller_g{version}"
        pairs = [(q, f"{value} days") for q in TRAIN]
        result = llm.train_adapter(adapter, pairs, out / "adapters", steps=args.steps)
        files = hash_files(result["path"])
        before = llm.generate(TRAIN[0], adapter=adapter)["text"]
        llm.reload_adapter(adapter, result["path"])
        after = llm.generate(TRAIN[0], adapter=adapter)["text"]
        if before != after: raise RuntimeError("Adapter reload changed output")
        result["reload_equal"] = True
        report["training"].append(result)
        artifact = registry.artifact("lora", {"adapter": adapter, "files": files, "base_sha256": base_hash,
            "training_sha256": digest(pairs), "transform_chain": [SemanticCompiler.VERSION, "assistant-only-lora-r8:v1"]},
            [cko["generation_key"]], principal="buyer")
        router.index(cko["generation_key"], artifact, principal="buyer")
        handles[adapter] = {"cko": cko, "artifact": artifact, "path": result["path"], "files": files}
        save()
        return handles[adapter]

    old = build(24, 1)
    # Wrong entity and wrong semantic role are real indexed, textual distractors.
    other = hard_record(99, 1)
    other.update(subject_id="supplier:muller-werke", subject_label="Müller Werke", trusted_aliases=[])
    other["source"]["record_id"] = "OTHER_SUPPLIER"
    other_cko = compiler.compile(other, principal="buyer")
    other_text = registry.artifact("text", {"fixture": "wrong entity"}, [other_cko["generation_key"]], principal="buyer")
    router.index(other_cko["generation_key"], other_text, principal="buyer")
    rule = hard_record(50, 1)
    rule["role"] = "RULE"
    rule["source"]["record_id"] = "RULE_92831"
    rule_cko = compiler.compile(rule, principal="buyer")
    rule_text = registry.artifact("text", {"fixture": "wrong semantic role"}, [rule_cko["generation_key"]], principal="buyer")
    router.index(rule_cko["generation_key"], rule_text, principal="buyer")
    report["router_training"] = router.fit([(q, old["cko"]["knowledge_key"]) for q in TRAIN] +
         [("What is the delivery time for X12 from Müller Werke?", other_cko["knowledge_key"])], principal="buyer")

    def infer(q):
        selected = router.select(q, principal="buyer", filters={"tags": ["lead-time"], "domain": "procurement", "type": "supplier_metric"})
        payload = registry.node(selected["artifact_key"])["payload"]["payload"]
        if registry.node(selected["artifact_key"])["kind"] != "lora": raise InvalidState("Selected artifact is not a neural Pod")
        verify_files(out / "adapters" / payload["adapter"], payload["files"])
        result = llm.generate(q, adapter=payload["adapter"])
        receipt = registry.commit(selected["snapshot"], result["text"])
        selected["snapshot"] = asdict(selected["snapshot"])
        return {"question": q, **result, "selection": selected, "receipt": receipt}

    for q in HELD_OUT:
        result = infer(q)
        result.update(phase="g1", expected="24 days", exact=result["text"].strip().lower() == "24 days")
        report["answers"].append(result)
    save()
    print(json.dumps({"phase": "g1", "correct": sum(x["exact"] for x in report["answers"])}), flush=True)
    previous_snapshot = registry.snapshot([old["artifact"]], principal="buyer")
    updated = build(18, 2)
    for q in HELD_OUT + TRAIN:
        result = infer(q)
        result.update(phase="g2", expected="18 days", exact=result["text"].strip().lower() == "18 days")
        report["answers"].append(result)
    def blocked(fn):
        try: fn()
        except InvalidState: return True
        return False
    report["lifecycle"]["old_commit_blocked"] = blocked(lambda: registry.commit(previous_snapshot, "24 days"))
    report["lifecycle"]["wrong_principal_blocked"] = blocked(lambda: router.select(HELD_OUT[0], principal="outsider"))
    pending = router.select(HELD_OUT[0], principal="buyer")["snapshot"]
    report["lifecycle"]["revocation_closure"] = registry.revoke(updated["cko"]["origin_keys"][0])
    report["lifecycle"]["revoked_commit_blocked"] = blocked(lambda: registry.commit(pending, "18 days"))
    report["lifecycle"]["revoked_question_paths"] = [blocked(lambda q=q: infer(q)) for q in HELD_OUT + TRAIN]
    restored = compiler.compile(hard_record(24, 3), principal="buyer")
    old_payload = registry.node(old["artifact"])["payload"]["payload"]
    restored_artifact = registry.artifact("lora", {**old_payload, "restored_from": old["artifact"]},
                    [restored["generation_key"], *old["cko"]["origin_keys"]], principal="buyer")
    router.index(restored["generation_key"], restored_artifact, principal="buyer")
    report["restored"] = infer(HELD_OUT[0])
    report["four_keys"] = {"origin_keys": registry.roots(restored_artifact), "knowledge_key": restored["knowledge_key"],
                          "generation_key": restored["generation_key"], "artifact_key": restored_artifact}
    report["canonical_knowledge_object"] = compiler.describe(restored["generation_key"])
    report["base_unchanged"] = llm.frozen_hash() == base_hash
    report["summary"] = {phase: {"correct": sum(a["exact"] for a in report["answers"] if a["phase"] == phase),
                               "n": sum(a["phase"] == phase for a in report["answers"])} for phase in ["g1", "g2"]}
    report["status"] = "completed"
    save()
    router.close()
    registry.close()
    print(json.dumps({"summary": report["summary"], "output": str(out)}), flush=True)


if __name__ == "__main__": main()
