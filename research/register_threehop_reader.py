"""Register the trained three-hop Qwen adapter with explicit lifecycle lineage."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.registry import Registry, InvalidState
from neural_pods.execution_manifest import ExecutionManifest


def sha(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter-run", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    adapter = a.adapter_run / "adapter_model.safetensors"
    if not adapter.exists(): raise FileNotFoundError(adapter)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    reg = Registry(a.output)
    try:
        origin = reg.origin("training", a.adapter_run.name, "1", {"adapter_sha256": sha(adapter)}, acl=("buyer",))
        source = reg.origin("knowledge", "supplier-three-hop", "1", {"facts": "transport+customs+warehouse"}, acl=("buyer",))
        generation = reg.publish("supplier:arrival:three_hop", {"hop_count": 3, "value_unit": "days"}, [source], principal="buyer")
        artifact = reg.artifact("lora", {"schema": "value-bearing-qwen-lora:v1", "adapter_sha256": sha(adapter),
            "adapter_path": str(adapter.resolve()), "generation_key": generation,
            "training_origin": origin, "activation_contract": "generation_and_training_lineage"},
            [generation, origin], principal="buyer")
        vector = reg.artifact("vector", {"embedding": [0.0, 1.0], "pod_type": "reasoning"}, [generation], principal="buyer")
        manifest = ExecutionManifest.build(reg, generation, [vector], reader_key=artifact, principal="buyer")
        active = reg.snapshot([artifact], "buyer")
        reg.revoke(origin)
        blocked = False
        try: reg.snapshot([artifact], "buyer")
        except InvalidState: blocked = True
        manifest_blocked = False
        try: manifest.validate(reg)
        except InvalidState: manifest_blocked = True
        result = {"schema":"three-hop-reader-registration:v1", "passed":blocked,
                  "origin_key":origin, "generation_key":generation, "artifact_key":artifact,
                  "adapter_sha256":sha(adapter), "activation_before_revocation":list(active.artifacts),
                  "revoked_activation_blocked":blocked, "manifest_generation_bound": manifest.reader_key == artifact,
                  "revoked_manifest_blocked":manifest_blocked}
        a.output.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2)); return 0 if blocked else 1
    finally: reg.close()


if __name__ == "__main__": raise SystemExit(main())
