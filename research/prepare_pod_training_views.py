"""Partition the mixed curriculum into model-safe training views.

One causal-LM trainer must never silently consume embedding, OCR, quantization,
or preference records. This creates deterministic per-track views and manifests.
"""
import json, shutil, sys
from collections import defaultdict
from pathlib import Path

src, out = map(Path, sys.argv[1:3])
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
rows_by_split = {}
for split in ("train", "dev", "test"):
    rows_by_split[split] = json.loads((src / "inputs" / f"{split}.json").read_text(encoding="utf-8"))

groups = defaultdict(lambda: defaultdict(list))
for split, rows in rows_by_split.items():
    for row in rows:
        groups[row["model_track"]][split].append(row)

specs = {
    "FunctionGemma-270M": {"objective": "tool_call_sft", "trainer": "causal_sft", "needs": ["schema_validator"]},
    "Qwen3-4B-SFT": {"objective": "grounded_reader_sft", "trainer": "causal_sft", "needs": ["chat_conversion", "citation_validator"]},
    "Qwen3-SFT": {"objective": "link_and_multihop_sft", "trainer": "causal_sft", "needs": ["chat_conversion", "provenance_validator"]},
    "Qwen3-GRPO": {"objective": "parallel_reasoning_rl", "trainer": "grpo", "needs": ["rollout_server", "reward_harness", "anti_hacking"]},
    "Zephyr-DPO": {"objective": "preference_optimization", "trainer": "dpo_or_orpo", "needs": ["chosen_rejected_pairs"]},
    "LoftQ-QAT": {"objective": "quantization_recipe", "trainer": "quantization", "needs": ["calibration_split", "unseen_holdout"]},
    "PaddleOCR-1B": {"objective": "vision_ocr", "trainer": "vision_sft", "needs": ["image_or_page_payload", "redaction_validator"]},
    "MoE-experts": {"objective": "expert_routing", "trainer": "router_sft_or_rl", "needs": ["capacity_metrics", "load_balance"]},
    "all": {"objective": "lifecycle_evaluation", "trainer": "symbolic_gate_plus_eval", "needs": ["frozen_protocol", "manifest_validator"]},
}

manifest = {"source": str(src), "dataset_version": "podmodel_v1", "views": {}}
for track, by_split in sorted(groups.items()):
    safe = track.lower().replace("/", "-").replace(" ", "-")
    view = out / safe / "inputs"
    view.mkdir(parents=True)
    for split in ("train", "dev", "test"):
        (view / f"{split}.json").write_text(json.dumps(by_split[split], ensure_ascii=False, indent=2), encoding="utf-8")
    spec = specs.get(track, {"objective": "unknown", "trainer": "manual_review", "needs": ["manual_review"]})
    manifest["views"][track] = {"path": str(view.parent), "rows": {s: len(by_split[s]) for s in ("train", "dev", "test")}, **spec}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({k: v["rows"] for k, v in manifest["views"].items()}))
