"""Build a modular Pod/model curriculum with stable provenance fields.

The source directory supplies the existing reader training config and protocol.
Each capability gets 50 examples per split so adapters can be retrained or
replaced independently without changing the canonical Pod identity.
"""
import hashlib, json, math, shutil, sys
from pathlib import Path

src, out = map(Path, sys.argv[1:3])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out)

CAPS = {
    "router_function": ("model", "FunctionGemma-270M", "Select the strict tool call for request {i}.", lambda i: {"tool": ["ann_search", "metadata_filter", "graph_neighbors"][i % 3], "strict_json": True, "deny_unknown_fields": True}),
    "reader_sft": ("reader", "Qwen3-4B-SFT", "Answer request {i} only from the attested evidence.", lambda i: {"answer_mode": "grounded", "cite": [f"artifact:A{i+1}"], "generation": f"g{(i % 4) + 1}"}),
    "pod_linking": ("transport", "Qwen3-SFT", "Verify remote Pod link {i} before using its result.", lambda i: {"transport": "ssh_tunnel", "peer": f"pod-peer-{i+1}", "verify": ["pod_id", "generation", "artifact_id", "acl"], "hop_budget": 3}),
    "multihop": ("research", "Qwen3-SFT", "Resolve the validated {hops}-hop chain for case {i}.", lambda i: {"hops": (i % 4) + 2, "require": ["origin_key", "knowledge_key", "generation_key", "artifact_key"], "reject_revoked": True}),
    "parallel_search": ("research", "Qwen3-GRPO", "Plan independent retrieval branches for case {i}.", lambda i: {"branches": ["ann_search", "bm25_search", "metadata_filter"], "parallel_calls": 3 + (i % 4), "join": "RRF", "barrier": "all_inputs_ready"}),
    "preference": ("alignment", "Zephyr-DPO", "Choose the preferred answer for comparison {i}.", lambda i: {"prefer": "current_grounded_compliant", "reject": "stale_unsupported_or_unprovenanced"}),
    "reasoning_rl": ("reasoning", "Qwen3-GRPO", "Score rollout {i} with correctness and anti-hacking checks.", lambda i: {"reward": {"correctness": 1.0, "evidence": 1.0, "latency": 0.5, "tool_efficiency": 0.5}, "anti_hacking": ["no_fake_citation", "no_reward_only_shortcut"]}),
    "quantization": ("model", "LoftQ-QAT", "Choose a quantization plan for model Pod {i}.", lambda i: {"method": "LoftQ" if i % 2 == 0 else "QAT", "calibration_split": "calibration_only", "holdout": "unseen", "metrics": ["task_accuracy", "KLD", "memory"]}),
    "vision_ocr": ("vision", "PaddleOCR-1B", "Create a privacy-safe OCR record for page {i}.", lambda i: {"text": "<redacted>", "page": i + 1, "bbox": [0, 0, 100, 30], "confidence": 0.99, "origin_key": f"src:scan:{i+1}", "redact_before_external": True}),
    "moe_routing": ("router", "MoE-experts", "Route request {i} to active experts without violating capability filters.", lambda i: {"experts": ["research", "reader", "math"], "hard_filter": "active_capability", "load_balance": True, "fallback": "local_reader"}),
    "lifecycle": ("lifecycle", "all", "Decide whether candidate Pod generation {i} may become active.", lambda i: {"transition": "candidate_to_active", "require": ["tests_pass", "manifest_hash", "attestation"], "block_if": ["revoked", "stale_generation", "acl_failure"]}),
    "evaluation": ("evaluation", "all", "Record frozen evaluation {i} for a Pod/model artifact.", lambda i: {"protocol_frozen": True, "metrics": ["recall_at_k", "exact_match", "latency_p95", "qps", "memory", "provenance_violations"], "compare": ["baseline", "previous_generation"]}),
}

def make(split):
    rows = []
    for cap, (pod_type, model_track, template, target_fn) in CAPS.items():
        for i in range(50):
            q = template.format(i=i + 1, hops=(i % 4) + 2)
            rows.append({
                "id": f"{split}:podmodel:{cap}:{i}", "task": cap,
                "pod_type": pod_type, "model_track": model_track, "language": "en",
                "question": q, "history": [], "evidence": None,
                "target": json.dumps(target_fn(i), separators=(",", ":")),
                "provenance": {"origin_key": f"src:synthetic:podmodel:{cap}:{i}", "dataset_version": "podmodel_v1", "source": "research_curriculum_generator"},
                "assessment": {"capability": cap, "pod_type": pod_type, "model_track": model_track, "example_index": i},
            })
    return rows

for split in ("train", "dev", "test"):
    p = out / "inputs" / f"{split}.json"
    p.write_text(json.dumps(make(split), ensure_ascii=False, indent=2), encoding="utf-8")

cfg = json.loads((out / "inputs" / "config.json").read_text())
t = cfg["training"]
protocol = json.loads((out / "protocol.json").read_text())
protocol.update({
    "status": "prepared_not_trained_pod_model_curriculum",
    "purpose": "Independent Pod/model tracks, remote links, lifecycle, quantization, OCR, MoE and evaluation",
    "rows": {s: len(make(s)) for s in ("train", "dev", "test")},
    "capability_count": len(CAPS), "examples_per_capability": {k: 50 for k in CAPS},
    "optimizer_updates_per_epoch": math.ceil(len(make("train")) / t["effective_batch_size"]),
    "optimizer_updates": math.ceil(len(make("train")) / t["effective_batch_size"]) * t["epochs"],
    "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "inputs").iterdir()},
    "model_tracks": sorted({v[1] for v in CAPS.values()}),
})
(out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
print(json.dumps(protocol["rows"]))
