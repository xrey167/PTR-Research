"""Train real LoRA weights, evaluate held-out questions, exercise lineage barriers."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics
import time

import psutil
import torch
from neural_pods.data import FACTS, UNRELATED, dataset_record, questions
from neural_pods.model import PodModel
from neural_pods.registry import Registry, InvalidState, digest, hash_files, verify_files
from neural_pods.routing import Router


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def correct(text, expected):
    # No substring scoring: 124 or '24 or 18 days' is not accepted as 24.
    normalized = text.strip().lower().rstrip(".! ")
    return normalized == expected.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if args.steps < 1: parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    process = psutil.Process()
    started = time.perf_counter()
    report = {"status": "running", "fixture": "3 fictional mutable facts; pilot, not a production RAG comparison",
              "config": vars(args) | {"output": str(out)}, "training": [], "evaluations": [], "lifecycle": {},
              "environment": {"platform": platform.platform(), "device": "cpu", "ram_bytes": psutil.virtual_memory().total,
                              "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "qdrant-client", "sentence-transformers"]}}}
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    report["models"] = manifest
    data = dataset_record(FACTS)
    write_json(out / "dataset.json", data)
    report["dataset_sha256"] = digest(data)
    write_json(out / "report.json", report)
    registry = Registry(out / "registry.sqlite3")
    llm = PodModel(manifest["qwen"]["path"], args.threads)
    base_hash = llm.frozen_hash()
    report["base_weights_sha256_before"] = base_hash
    router = Router(manifest["encoder"]["path"], FACTS, registry, out)
    handles = {}
    training_pairs = lambda fs: [(q, f.answer) for f in fs for q in questions(f, "train")]

    def persist():
        report["elapsed_s"] = time.perf_counter() - started
        report["process_rss_bytes"] = process.memory_info().rss
        write_json(out / "report.json", report)

    def train(fact, name):
        origin = registry.origin("fixture:supplier-system", fact.key, str(fact.generation), fact.semantic())
        knowledge = registry.publish(fact.key, fact.semantic(), [origin])
        result = llm.train_adapter(name, training_pairs([fact]), out / "adapters", steps=args.steps)
        files = hash_files(result["path"])
        pod = registry.artifact("lora", {"adapter": name, "files": files, "base_sha256": base_hash,
                   "base_revision": manifest["qwen"]["revision"], "training_sha256": digest(training_pairs([fact]))}, [knowledge])
        # Verify saved weights by actual unloading/reloading and greedy generation.
        q = questions(fact, "validation")[0]
        before = llm.generate(q, adapter=name)["text"]
        verify_files(result["path"], files)
        llm.reload_adapter(name, result["path"])
        after = llm.generate(q, adapter=name)["text"]
        result["save_reload_same_output"] = before == after
        result["validation"] = {"question": q, "text": after, "expected": fact.answer, "exact": correct(after, fact.answer)}
        if before != after: raise RuntimeError("Saved adapter failed reload parity")
        report["training"].append(result)
        handle = {"fact": fact, "origin": origin, "knowledge": knowledge, "pod": pod,
                  "adapter": name, "files": files, "path": result["path"]}
        handles[name] = handle
        router.upsert(fact, knowledge, pod, name)
        persist()
        return handle

    for i, fact in enumerate(FACTS):
        train(fact, f"pod{i}_g1")
    all_result = llm.train_adapter("all_facts", training_pairs(FACTS), out / "adapters", steps=args.steps * 2)
    report["training"].append(all_result)
    all_artifact = registry.artifact("lora", {"adapter": "all_facts", "files": hash_files(all_result["path"]),
            "base_sha256": base_hash}, [h["knowledge"] for h in handles.values()])
    report["monolithic_artifact"] = all_artifact
    report["base_weights_sha256_after_training"] = llm.frozen_hash()
    if report["base_weights_sha256_after_training"] != base_hash:
        raise RuntimeError("Frozen base weights changed")
    persist()

    def infer(question, mode):
        begin = time.perf_counter()
        if mode in {"base", "always_loaded"}:
            output = llm.generate(question, adapter="all_facts" if mode == "always_loaded" else None)
            output["total_s"] = time.perf_counter() - begin
            return output
        learned = mode == "routed_pod"
        selected = router.select(question, learned=learned)
        deps = [selected["vector_artifact"]]
        if learned:
            handle = handles[selected["adapter"]]
            verify_files(handle["path"], handle["files"])
            deps += [selected["pod_artifact"], router.artifact]
        snapshot = registry.snapshot(deps)
        output = llm.generate(question, adapter=selected["adapter"] if learned else None,
                              evidence=None if learned else selected["evidence"])
        receipt = registry.commit(snapshot, output["text"])
        output.update(total_s=time.perf_counter() - begin, selected_key=selected["key"],
                      selected_generation=selected["generation"], similarity=selected["similarity"],
                      router_score=selected["router_score"], receipt=receipt["answer_id"])
        return output

    # Test split is first touched only after all initial adapters are frozen.
    for mode in ["base", "qdrant_rag", "always_loaded", "routed_pod"]:
        for fact in FACTS:
            for q in questions(fact, "test"):
                record = {"phase": "initial", "mode": mode, "question": q, "key": fact.key, "expected": fact.answer}
                try:
                    record.update(infer(q, mode))
                    record["exact"] = correct(record["text"], fact.answer)
                except InvalidState as exc:
                    record.update(text="", exact=False, blocked=str(exc))
                report["evaluations"].append(record)
        print(json.dumps({"event": "evaluation", "mode": mode,
              "correct": sum(r["exact"] for r in report["evaluations"] if r["mode"] == mode)}), flush=True)
        persist()

    # Capability interference is a small diagnostic, not a broad capability benchmark.
    report["unrelated"] = []
    for question, expected in UNRELATED:
        for adapter in [None, "all_facts", "pod0_g1"]:
            result = llm.generate(question, adapter=adapter)
            report["unrelated"].append({"question": question, "expected": expected,
                 "adapter": adapter, **result, "exact": correct(result["text"], expected)})

    # Controlled update while a previous snapshot / cache still exists.
    old = handles["pod0_g1"]
    old_snapshot = registry.snapshot([old["pod"]])
    cache = registry.artifact("cache", {"text": "24 days"}, [old["pod"]])
    updated_fact = replace(FACTS[0], value=18, generation=2)
    updated = train(updated_fact, "pod0_g2")
    lifecycle = report["lifecycle"]
    def blocked(fn):
        try: fn()
        except InvalidState: return True
        return False
    lifecycle["stale_snapshot_rejected_after_update"] = blocked(lambda: registry.commit(old_snapshot, "24 days"))
    lifecycle["old_cache_rejected_after_update"] = blocked(lambda: registry.snapshot([cache]))
    lifecycle["mixed_generations_rejected"] = blocked(lambda: registry.snapshot([old["pod"], updated["pod"]]))
    for q in questions(updated_fact, "train") + questions(updated_fact, "test"):
        try: result = infer(q, "routed_pod")
        except InvalidState as exc: result = {"text": "", "blocked": str(exc)}
        report["evaluations"].append({"phase": "updated", "mode": "routed_pod", "question": q,
               "key": updated_fact.key, "expected": updated_fact.answer, **result,
               "exact": correct(result["text"], updated_fact.answer)})
    # Raw old weights remain a deliberate unguarded ablation.
    lifecycle["unguarded_old_adapter_after_update"] = llm.generate(questions(FACTS[0], "test")[0], adapter="pod0_g1")
    stale = registry.snapshot([updated["pod"]])
    affected = registry.revoke(updated["origin"])
    lifecycle["revocation_closure"] = affected
    lifecycle["inflight_commit_rejected_after_revoke"] = blocked(lambda: registry.commit(stale, "18 days"))
    lifecycle["revoked_paraphrases_blocked"] = [blocked(lambda q=q: infer(q, "routed_pod"))
                     for q in questions(updated_fact, "train") + questions(updated_fact, "test")]
    lifecycle["unrelated_pod_survives"] = not blocked(lambda: registry.snapshot([handles["pod1_g1"]["pod"]]))
    lifecycle["unrelated_answer_after_revoke"] = infer(questions(FACTS[1], "test")[0], "routed_pod")
    # Restore creates generation 3; old generations never become valid again.
    restored = replace(FACTS[0], generation=3)
    restore_origin = registry.origin("fixture:restore", restored.key, "3", restored.semantic())
    restored_k = registry.publish(restored.key, restored.semantic(), [restore_origin])
    restored_pod = registry.artifact("lora", {"adapter": old["adapter"], "files": old["files"],
        "base_sha256": base_hash, "restored_from_weights": old["pod"]}, [restored_k, old["origin"]])
    # The restored artifact depends on unchanged original evidence, not a revoked generation.
    router.upsert(restored, restored_k, restored_pod, old["adapter"])
    lifecycle["restored_generation"] = registry.node(restored_k)["payload"]["generation"]
    lifecycle["restored_answer"] = infer(questions(restored, "test")[0], "routed_pod")
    lifecycle["old_identity_still_rejected"] = blocked(lambda: registry.snapshot([old["pod"]]))

    # Multi-Pod question uses two guarded neural reads, followed by base-model composition.
    a = infer(questions(FACTS[1], "test")[0], "routed_pod")
    b = infer(questions(FACTS[2], "test")[0], "routed_pod")
    pair_snapshot = registry.snapshot([a["receipt"], b["receipt"]])
    composed = llm.generate("What is the sum of these two delivery times? Reply with only the number and days.",
                   evidence=f"First delivery: {a['text']}. Second delivery: {b['text']}.")
    registry.commit(pair_snapshot, composed["text"])
    report["multi_pod"] = {"method": "two neural reads + textual composition; no adapter fusion",
                          "first": a, "second": b, "result": composed, "expected": "53 days",
                          "exact": correct(composed["text"], "53 days")}
    report["summary"] = {}
    for mode in ["base", "qdrant_rag", "always_loaded", "routed_pod"]:
        records = [r for r in report["evaluations"] if r["phase"] == "initial" and r["mode"] == mode]
        times = [r["total_s"] for r in records if "total_s" in r]
        report["summary"][mode] = {"correct": sum(r["exact"] for r in records), "n": len(records),
             "mean_latency_s": statistics.mean(times) if times else None,
             "median_latency_s": statistics.median(times) if times else None,
             "mean_input_tokens": statistics.mean(r["input_tokens"] for r in records if "input_tokens" in r),
             "tokens_per_s": sum(r.get("output_tokens", 0) for r in records) / sum(times) if times else None}
    report["base_weights_sha256_final"] = llm.frozen_hash()
    report["base_unchanged"] = report["base_weights_sha256_final"] == base_hash
    report["adapter_bytes"] = sum(p.stat().st_size for p in (out / "adapters").rglob("*") if p.is_file())
    report["status"] = "completed"
    persist()
    router.close()
    registry.close()
    print(json.dumps({"event": "completed", "output": str(out), "summary": report["summary"],
                      "elapsed_s": report["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
