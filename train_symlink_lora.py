"""Train actual Qwen link/cluster LoRAs and prove stable dereferencing across edits."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
from sentence_transformers import SentenceTransformer
from neural_pods.dragonfly import DragonflyRouter, alias_training_questions
from neural_pods.model import PodModel
from neural_pods.registry import Registry, InvalidState, hash_files, verify_files, digest
from neural_pods.symlink import NeuralSymlinks
from run_semantic_experiment import TRAIN, HELD_OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--link-steps", type=int, default=48)
    parser.add_argument("--link-lr", type=float, default=0.0003)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    old = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if old["status"] != "completed": raise ValueError("Use a completed source run")
    out.mkdir(parents=True, exist_ok=False)
    src = sqlite3.connect((source / "registry.sqlite3").as_uri() + "?mode=ro", uri=True)
    dst = sqlite3.connect(out / "registry.sqlite3")
    src.backup(dst); src.close(); dst.close()
    for folder in ("qdrant", "adapters"): shutil.copytree(source / folder, out / folder)
    manifest = {"status": "running", "models": old["models"], "base_weights_sha256": old["base_weights_sha256"],
                "router_kind": "dragonfly", "embedded_links": True, "copied_from": str(source)}
    report = {"status": "running", "training": [], "link_predictions": [], "answers": [], "fictional_data": True}
    started = time.perf_counter()
    def save():
        report["elapsed_s"] = time.perf_counter() - started
        (out / "symlink-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    reg = Registry(out / "registry.sqlite3")
    info = manifest["models"]["encoder"]
    encoder = SentenceTransformer(info["path"], device="cpu", local_files_only=True)
    router = DragonflyRouter(reg, encoder, out, encoder_id=info["repo"] + "@" + info["revision"])
    links = NeuralSymlinks(reg)
    try:
        router.load_weights()
        model = PodModel(manifest["models"]["qwen"]["path"])
        assert model.frozen_hash() == manifest["base_weights_sha256"]
        items = []
        for item in router.bindings():
            try: reg.snapshot([item["artifact_key"]], "buyer")
            except InvalidState: continue
            items.append(item)
        items.sort(key=lambda i: (i["node"]["semantic"]["role"], i["node"]["semantic"]["subject"]))
        assert len(items) == 3, "Expected two current facts and one rule"
        other = lambda q: re.sub(r"Müller GmbH|Muller GmbH|Mueller GmbH|Müller|Muller|Mueller", "Müller Werke", q)
        for item in items:
            semantic = item["node"]["semantic"]
            key = item["node"]["knowledge_key"]
            b = links.register(key, "buyer")
            prompts = [other(q) for q in TRAIN] if semantic["subject"] == "supplier:muller-werke" else TRAIN
            _, questions = alias_training_questions(semantic, prompts)
            pairs = [(q, links.target_text(key)) for q in questions]
            adapter = f"semantic_link_{b['slot']}"
            trained = model.train_adapter(adapter, pairs, out / "adapters", steps=args.link_steps, lr=args.link_lr, task="link")
            files = hash_files(trained["path"])
            artifact = links.attach(key, {"adapter": adapter, "files": files, "base_sha256": manifest["base_weights_sha256"],
                         "training_sha256": digest(pairs), "transform_chain": ["semantic-link:v1", "qwen-link-lora:r8"]}, pairs, "buyer")
            before = model.generate(questions[0], adapter=adapter, task="link")["text"]
            model.reload_adapter(adapter, trained["path"])
            after = model.generate(questions[0], adapter=adapter, task="link")["text"]
            report["training"].append({**trained, "artifact_key": artifact, "knowledge_key": key,
                "target": links.target_text(key), "reload_equal": before == after, "training_example_count": len(pairs)})
            save()
            assert before == after == links.target_text(key), (before, after, links.target_text(key))
            evaluation = [other(q) for q in HELD_OUT] if semantic["subject"] == "supplier:muller-werke" else HELD_OUT
            for q in evaluation:
                generated = model.generate(q, adapter=adapter, task="link")
                result = {"question": q, "role": semantic["role"], "knowledge_key": key,
                          "expected": links.target_text(key), **generated, "correct": generated["text"] == links.target_text(key)}
                report["link_predictions"].append(result)
                save()
                assert result["correct"], result
                resolved = links.resolve(generated["text"], expected_knowledge_key=key, principal="buyer")
                assert resolved["generation_key"] == item["generation_key"]
        main = items[0]
        key = main["node"]["knowledge_key"]
        assert main["node"]["semantic"]["subject"] == "supplier:muller"
        stable = links.active_binding(key, "buyer")
        link_payload = reg.node(stable["adapter_key"])["payload"]["payload"]
        link_hashes = hash_files(out / "adapters" / link_payload["adapter"])
        fact_cluster = stable["cluster_key"]
        report["cluster_members"] = links.cluster_members(fact_cluster, "buyer")
        assert len(report["cluster_members"]) == 2
        assert links.active_binding(items[2]["node"]["knowledge_key"], "buyer")["cluster_key"] != fact_cluster
        loaded_answers = set()
        def infer(q):
            t = time.perf_counter()
            first = router.select(q, principal="buyer")
            b = links.active_binding(first["knowledge_key"], "buyer")
            lp = reg.node(b["adapter_key"])["payload"]["payload"]
            verify_files(out / "adapters" / lp["adapter"], lp["files"])
            predicted = model.generate(q, adapter=lp["adapter"], task="link")
            resolved = links.resolve(predicted["text"], expected_knowledge_key=first["knowledge_key"], principal="buyer")
            current = router.select(q, principal="buyer")
            assert current["generation_key"] == resolved["generation_key"]
            p = reg.node(current["artifact_key"])["payload"]["payload"]
            verify_files(out / "adapters" / p["adapter"], p["files"])
            if p["adapter"] not in loaded_answers and p["adapter"] not in model.model.peft_config:
                model.model.load_adapter(out / "adapters" / p["adapter"], adapter_name=p["adapter"], is_trainable=False)
            loaded_answers.add(p["adapter"])
            snapshot = reg.snapshot([*current["snapshot"].artifacts, *resolved["snapshot"].artifacts], "buyer")
            answer = model.generate(q, adapter=p["adapter"])
            receipt = reg.commit(snapshot, answer["text"])
            return {"question": q, "prediction": predicted["text"], "link_artifact": b["adapter_key"],
                    "generation_key": current["generation_key"], "cluster_key": resolved["cluster_key"],
                    "text": answer["text"], "total_s": time.perf_counter() - t, "receipt": receipt}
        for q in HELD_OUT:
            result = infer(q); result["expected"] = "24 days"
            report["answers"].append(result); save()
            assert result["text"] == "24 days", result
        pending = links.resolve(links.target_text(key), expected_knowledge_key=key, principal="buyer")["snapshot"]
        # New factual value and answer adapter; no new link training.
        semantic = deepcopy(main["node"]["semantic"])
        semantic["object"]["value"] = 18
        origin = reg.origin("fixture:sap", "PO_HISTORY_92831", "symlink-update-18", {"lead_time_days": 18}, acl=["buyer"])
        life = {**main["node"]["lifecycle"], "source": origin}
        generation = reg.publish(key, semantic, [origin], principal="buyer", acl=["buyer"], lifecycle=life)
        answer_pairs = [(q, "18 days") for q in TRAIN]
        trained = model.train_adapter("symlink_answer_18", answer_pairs, out / "adapters", steps=args.steps)
        report["answer_training"] = trained; save()
        pod = reg.artifact("lora", {"adapter": "symlink_answer_18", "files": hash_files(trained["path"]),
                    "base_sha256": manifest["base_weights_sha256"], "training_sha256": digest(answer_pairs)}, [generation], "buyer")
        router.index(generation, pod, "buyer")
        router.fit_pod(generation, TRAIN, [other(q) for q in TRAIN], negative_aliases=["Müller Werke"], principal="buyer")
        assert links.register(key, "buyer")["adapter_key"] == stable["adapter_key"]
        for q in HELD_OUT:
            result = infer(q); result["expected"] = "18 days"
            report["answers"].append(result); save()
            assert result["text"] == "18 days", result
            assert result["link_artifact"] == stable["adapter_key"] and result["generation_key"] == generation
        assert hash_files(out / "adapters" / link_payload["adapter"]) == link_hashes
        try: reg.commit(pending, "24 days")
        except InvalidState: old_blocked = True
        else: old_blocked = False
        clone = Registry(":memory:")
        try:
            reg.db.backup(clone.db)
            clone_links = NeuralSymlinks(clone)
            snapshot = clone_links.resolve(links.target_text(key), expected_knowledge_key=key, principal="buyer")["snapshot"]
            clone.revoke(origin)
            try: clone.commit(snapshot, "18 days")
            except InvalidState: revoked_blocked = True
            else: revoked_blocked = False
            try: clone_links.resolve(links.target_text(key), expected_knowledge_key=key, principal="buyer")
            except InvalidState: link_blocked = True
            else: link_blocked = False
            assert len(clone_links.cluster_members(fact_cluster, "buyer")) == 1
        finally: clone.close()
        report["lifecycle"] = {"old_commit_blocked": old_blocked, "revoked_commit_blocked": revoked_blocked,
            "revoked_target_blocked": link_blocked, "same_link_adapter_after_update": True, "same_cluster_after_update": True}
        report["base_unchanged"] = model.frozen_hash() == manifest["base_weights_sha256"]
        assert report["base_unchanged"] and all(report["lifecycle"].values())
        report["status"] = "passed"; manifest["status"] = "completed"
        print(json.dumps({"status": "passed", "link_predictions": len(report["link_predictions"]),
                          "answers": len(report["answers"]), "output": str(out)}), flush=True)
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save(); router.close(); reg.close()


if __name__ == "__main__": main()
