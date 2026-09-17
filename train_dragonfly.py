"""Train and evaluate local Pod addresses, reusing real saved LoRA weights.

Creates an isolated copy of a completed semantic run; never rewrites the source.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from neural_pods.dragonfly import DragonflyRouter
from neural_pods.model import PodModel
from neural_pods.registry import Registry, InvalidState, verify_files
from run_semantic_experiment import TRAIN, HELD_OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    original = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if original["status"] != "completed": raise ValueError("Use a completed semantic run")
    out.mkdir(parents=True, exist_ok=False)
    src = sqlite3.connect((source / "registry.sqlite3").as_uri() + "?mode=ro", uri=True)
    dest = sqlite3.connect(out / "registry.sqlite3")
    src.backup(dest)
    src.close()
    dest.close()
    for folder in ("qdrant", "adapters"): shutil.copytree(source / folder, out / folder)
    manifest = {**original, "status": "running", "router_kind": "dragonfly", "copied_from": str(source)}
    (out / "report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"status": "running", "source": str(source), "training": [], "routing": [], "answers": [], "blocked": [],
              "scope": "New address-vector training; unchanged saved LoRA weights; synthetic procurement fixture"}
    started = time.perf_counter()
    def save():
        result["elapsed_s"] = time.perf_counter() - started
        (out / "dragonfly-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    registry = Registry(out / "registry.sqlite3")
    info = original["models"]["encoder"]
    encoder = SentenceTransformer(info["path"], device="cpu", local_files_only=True)
    encoder_id = info["repo"] + "@" + info["revision"]
    router = DragonflyRouter(registry, encoder, out, encoder_id=encoder_id)
    try:
        items = []
        for item in router.bindings():
            try: registry.snapshot([item["vector_key"], item["artifact_key"]], "buyer")
            except InvalidState: continue
            if item["node"]["semantic"]["role"] == "FACT": items.append(item)
        items.sort(key=lambda i: i["node"]["semantic"]["subject"])
        if [i["node"]["semantic"]["subject"] for i in items] != ["supplier:muller", "supplier:muller-werke"]:
            raise ValueError("This evaluation fixture requires current Muller and Muller Werke facts")
        def other_company(q):
            return re.sub(r"Müller GmbH|Muller GmbH|Mueller GmbH|Müller|Muller|Mueller", "Müller Werke", q)
        # Independently authored fictional address catalogue. Expand training
        # templates across spellings, never import evaluation sentences.
        names = ["Müller GmbH", "Müller", "Muller GmbH", "Muller", "Mueller GmbH", "Mueller"]
        questions = [TRAIN, [other_company(q) for q in TRAIN]]
        fresh = ["Give the current lead time for supplier Mueller and part X12.",
                 "State the recorded delivery duration for X12 from Muller GmbH.",
                 "Tell me the procurement delay for Mueller GmbH and component X12."]
        evaluations = [HELD_OUT + fresh, [other_company(q) for q in HELD_OUT + fresh]]
        assert not set(sum(questions, [])) & set(sum(evaluations, []))
        result["training_questions"] = questions
        for i, item in enumerate(items):
            metrics = router.fit_pod(item["generation_key"], questions[i], questions[1-i], principal="buyer",
                                     negative_aliases=["Müller Werke"] if i == 0 else names)
            result["training"].append({"knowledge_key": item["node"]["knowledge_key"], **metrics})
        save()
        # Rank both candidates with identical entity/ACL/time/tag features to
        # measure learned semantic discrimination without hard-filter shortcuts.
        for expected, held_out in enumerate(evaluations):
            for q in held_out:
                vector = encoder.encode(q, normalize_embeddings=True)
                scores, baseline, z_only = [], [], []
                for item in items:
                    embedding = registry.node(item["vector_key"])["payload"]["payload"]["embedding"]
                    similarity = float(np.dot(vector, embedding))
                    scores.append(router.learned_score(vector, [similarity] + [1.] * 8, item))
                    baseline.append(similarity)
                    z = router.addresses[item["generation_key"]][1].z.detach().numpy()
                    z_only.append(float(np.dot(vector, z / np.linalg.norm(z))))
                result["routing"].append({"question": q, "expected_index": expected, "scores": scores,
                    "correct": int(np.argmax(scores)) == expected, "cosine_correct": int(np.argmax(baseline)) == expected,
                    "z_only_correct": int(np.argmax(z_only)) == expected})
        # Verify exact score after closing/reopening the persistent router.
        before = router.select(HELD_OUT[0], principal="buyer")
        router.close()
        router = DragonflyRouter(registry, encoder, out, encoder_id=encoder_id)
        router.load_weights()
        # Recognition from learned vectors alone, after activation in a new
        # router instance; no resolve_query or trusted-alias matching involved.
        result["alias_recognition"] = []
        for subject, aliases in [("supplier:muller", names), ("supplier:muller-werke", ["Müller Werke"])]:
            for alias in aliases:
                recognized = router.recognize_alias(alias, principal="buyer")
                result["alias_recognition"].append({"alias": alias, "subject": recognized["subject"],
                    "score": recognized["score"], "correct": recognized["subject"] == subject})
        after = router.select(HELD_OUT[0], principal="buyer")
        result["reload_equal"] = before["score"] == after["score"] and before["representation_key"] == after["representation_key"]
        model = PodModel(original["models"]["qwen"]["path"])
        assert model.frozen_hash() == original["base_weights_sha256"]
        loaded = set()
        for q in HELD_OUT:
            selected = router.select(q, principal="buyer")
            payload = registry.node(selected["artifact_key"])["payload"]["payload"]
            adapter = payload["adapter"]
            path = out / "adapters" / adapter
            verify_files(path, payload["files"])
            if adapter not in loaded:
                model.model.load_adapter(path, adapter_name=adapter, is_trainable=False)
                loaded.add(adapter)
            generated = model.generate(q, adapter=adapter)
            receipt = registry.commit(selected["snapshot"], generated["text"])
            selected["snapshot"] = asdict(selected["snapshot"])
            result["answers"].append({"question": q, "text": generated["text"], "correct": generated["text"] == "24 days",
                                      "selection": selected, "receipt": receipt})
            save()
        for q, principal in [("What is the delivery lead time for X99 from Mueller GmbH?", "buyer"),
                             ("How long has Mueller GmbH existed?", "buyer"), (HELD_OUT[0], "outsider")]:
            try: router.select(q, principal=principal)
            except InvalidState as exc: result["blocked"].append({"question": q, "principal": principal, "blocked": True, "reason": str(exc)})
            else: result["blocked"].append({"question": q, "principal": principal, "blocked": False})
        clone = Registry(":memory:")
        try:
            registry.db.backup(clone.db)
            selected = router.select(HELD_OUT[0], principal="buyer")
            dependent = clone.commit(selected["snapshot"], "24 days")["answer_id"]
            closure = clone.revoke(selected["generation_key"])
            try: clone.commit(selected["snapshot"], "24 days")
            except InvalidState: rejected = True
            else: rejected = False
            other_rep = router.addresses[items[1]["generation_key"]][0]
            clone.snapshot([other_rep], principal="buyer")
            result["revocation"] = {"representation_in_closure": selected["representation_key"] in closure,
                "answer_in_closure": dependent in closure, "pending_commit_blocked": rejected, "other_representation_valid": True}
        finally: clone.close()
        result["base_unchanged"] = model.frozen_hash() == original["base_weights_sha256"]
        result["summary"] = {"routing_correct": sum(x["correct"] for x in result["routing"]),
            "cosine_correct": sum(x["cosine_correct"] for x in result["routing"]),
            "z_only_correct": sum(x["z_only_correct"] for x in result["routing"]), "routing_n": len(result["routing"]),
            "answers_correct": sum(x["correct"] for x in result["answers"]), "answers_n": len(result["answers"])}
        assert result["reload_equal"] and result["base_unchanged"]
        assert all(x["blocked"] for x in result["blocked"]) and all(result["revocation"].values())
        assert all(x["correct"] for x in result["answers"])
        assert all(x["correct"] for x in result["routing"])
        assert all(x["correct"] for x in result["alias_recognition"])
        result["status"] = "passed"
        manifest["status"] = "completed"
        (out / "report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        save()
        router.close()
        registry.close()
    print(json.dumps({"status": result["status"], "summary": result["summary"], "output": str(out)}, indent=2))


if __name__ == "__main__": main()
