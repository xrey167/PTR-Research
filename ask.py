"""Run a guarded neural answer from saved, verified model/adapter artifacts."""
import argparse
import json
from pathlib import Path
from neural_pods.model import PodModel
from neural_pods.registry import Registry, InvalidState, verify_files
from neural_pods.routing import Router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["pod", "rag"], default="pod")
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "completed":
        raise RuntimeError("Use a completed run")
    registry = Registry(run / "registry.sqlite3")
    llm = PodModel(report["models"]["qwen"]["path"])
    base_hash = llm.frozen_hash()
    if base_hash != report["base_weights_sha256_before"]:
        raise InvalidState("Base model does not match experiment")
    router = Router.load(report["models"]["encoder"]["path"], registry, run)
    try:
        selected = router.select(args.question, learned=args.mode == "pod")
        dependencies = [selected["vector_artifact"]]
        adapter = None
        if args.mode == "pod":
            payload = registry.node(selected["pod_artifact"])["payload"]["payload"]
            if payload["base_sha256"] != base_hash:
                raise InvalidState("Pod is incompatible with base model")
            adapter = payload["adapter"]
            directory = (run / "adapters" / adapter).resolve()
            if directory.parent != (run / "adapters").resolve():
                raise InvalidState("Invalid adapter path")
            verify_files(directory, payload["files"])
            llm.model.load_adapter(directory, adapter_name=adapter, is_trainable=False)
            dependencies += [selected["pod_artifact"], router.artifact]
        snapshot = registry.snapshot(dependencies)
        if args.mode == "rag":
            # Reuse the exact extraction implementation exercised by compare_rag.py.
            from compare_rag import generate, messages
            comparison = run / "rag_comparison.json"
            style = json.loads(comparison.read_text(encoding="utf-8"))["selected_prompt"] if comparison.exists() else "few_shot"
            output = generate(llm, messages(style, args.question, selected["evidence"]))
        else:
            output = llm.generate(args.question, adapter=adapter)
        receipt = registry.commit(snapshot, output["text"])
        print(json.dumps({**output, "knowledge_key": selected["key"], "generation": selected["generation"],
                          "receipt": receipt, "origin_keys": registry.roots(receipt["answer_id"])}, indent=2, ensure_ascii=False))
    except InvalidState as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
    finally:
        router.close()
        registry.close()


if __name__ == "__main__":
    main()
