"""Query the persisted typed semantic experiment using actual LoRA inference."""
import argparse
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from neural_pods.registry import Registry, InvalidState, verify_files
from neural_pods.semantic_routing import SemanticRouter
from neural_pods.dragonfly import DragonflyRouter
from neural_pods.model import PodModel
from neural_pods.symlink import NeuralSymlinks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("question")
    parser.add_argument("--principal", default="buyer", help="Local test identity; not an authentication mechanism")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--domain")
    parser.add_argument("--language")
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "completed": raise RuntimeError("Use a completed run")
    registry = Registry(run / "registry.sqlite3")
    encoder = SentenceTransformer(report["models"]["encoder"]["path"], device="cpu", local_files_only=True)
    if report.get("router_kind") == "dragonfly":
        encoder_info = report["models"]["encoder"]
        router = DragonflyRouter(registry, encoder, run, encoder_id=encoder_info["repo"] + "@" + encoder_info["revision"])
    else:
        router = SemanticRouter(registry, encoder, run)
    try:
        router.load_weights()
        selected = router.select(args.question, principal=args.principal,
                    filters={"tags": args.tag, "domain": args.domain, "language": args.language})
        node = registry.node(selected["artifact_key"])
        if node["kind"] != "lora": raise InvalidState("Candidate contains text, not a trained neural Pod")
        payload = node["payload"]["payload"]
        path = (run / "adapters" / payload["adapter"]).resolve()
        if path.parent != (run / "adapters").resolve(): raise InvalidState("Invalid adapter path")
        verify_files(path, payload["files"])
        model = PodModel(report["models"]["qwen"]["path"])
        if model.frozen_hash() != payload["base_sha256"]: raise InvalidState("Incompatible base weights")
        embedded = None
        if report.get("embedded_links"):
            links = NeuralSymlinks(registry)
            binding = links.active_binding(selected["knowledge_key"], args.principal)
            lp = registry.node(binding["adapter_key"])["payload"]["payload"]
            link_path = (run / "adapters" / lp["adapter"]).resolve()
            if link_path.parent != (run / "adapters").resolve(): raise InvalidState("Invalid link adapter path")
            verify_files(link_path, lp["files"])
            if lp["base_sha256"] != payload["base_sha256"]: raise InvalidState("Incompatible link base weights")
            model.model.load_adapter(link_path, adapter_name=lp["adapter"], is_trainable=False)
            prediction = model.generate(args.question, adapter=lp["adapter"], task="link")
            resolved = links.resolve(prediction["text"], expected_knowledge_key=selected["knowledge_key"], principal=args.principal)
            current = router.select(args.question, principal=args.principal,
                        filters={"tags": args.tag, "domain": args.domain, "language": args.language})
            if current["generation_key"] != resolved["generation_key"]:
                raise InvalidState("Link target changed during resolution")
            current["snapshot"] = registry.snapshot([*current["snapshot"].artifacts, *resolved["snapshot"].artifacts], args.principal)
            selected = current
            node = registry.node(selected["artifact_key"])
            if node["kind"] != "lora": raise InvalidState("Current target has no neural answer Pod")
            payload = node["payload"]["payload"]
            path = (run / "adapters" / payload["adapter"]).resolve()
            if path.parent != (run / "adapters").resolve(): raise InvalidState("Invalid current adapter path")
            verify_files(path, payload["files"])
            if payload["base_sha256"] != lp["base_sha256"]: raise InvalidState("Current Pod base mismatch")
            embedded = {"model_prediction": prediction["text"], "cluster_key": resolved["cluster_key"],
                        "link_artifact_key": binding["adapter_key"], "generation_key": resolved["generation_key"]}
        model.model.load_adapter(path, adapter_name=payload["adapter"], is_trainable=False)
        generated = model.generate(args.question, adapter=payload["adapter"])
        receipt = registry.commit(selected["snapshot"], generated["text"])
        print(json.dumps({"status": "committed", **generated, "knowledge_key": selected["knowledge_key"],
                "generation": selected["generation"], "generation_key": selected["generation_key"],
                "artifact_key": selected["artifact_key"], "origin_keys": registry.roots(selected["artifact_key"]),
                "receipt": receipt, "query_semantics": selected["query_semantics"],
                "routing_mode": selected.get("routing_mode", "semantic-mlp"),
                "representation_key": selected.get("representation_key"),
                "trained_aliases": (selected.get("alias_training") or {}).get("aliases", []),
                "embedded_link": embedded}, ensure_ascii=False, indent=2))
    except InvalidState as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False))
    finally:
        router.close()
        registry.close()


if __name__ == "__main__": main()
