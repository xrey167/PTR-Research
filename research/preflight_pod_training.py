"""Static preflight for a Pod/model training run; exits nonzero on blockers."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
cfg = json.loads((root / "inputs" / "config.json").read_text(encoding="utf-8"))
rows = json.loads((root / "inputs" / "train.json").read_text(encoding="utf-8"))
tracks = sorted({r["model_track"] for r in rows})
blockers, warnings = [], []
if len(tracks) > 1:
    blockers.append("mixed model tracks in one causal-LM dataset; use prepare_pod_training_views.py")
if cfg.get("model_path") in {"models/qwen3b", "/home/xrey/neural-pods/models/qwen3b"}:
    warnings.append("config points at qwen3b/Qwen2.5-3B while this curriculum names Qwen3 tracks; pin exact base per view")
if cfg.get("training", {}).get("world_size") == 1:
    warnings.append("world_size=1; do not assume both GPUs are used")
if cfg.get("training", {}).get("assistant_only_loss") and not all("messages" in r for r in rows):
    blockers.append("assistant_only_loss requires chat/message conversion; raw question/target rows are not trainer-ready")
if {"PaddleOCR-1B", "LoftQ-QAT"} & set(tracks):
    blockers.append("vision/quantization rows require dedicated pipelines, not the reader causal-LM trainer")
print(json.dumps({"tracks": tracks, "blockers": blockers, "warnings": warnings}, indent=2))
if blockers:
    raise SystemExit(2)
