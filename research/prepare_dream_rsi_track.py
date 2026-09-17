"""Create a replay-simulator track inspired by Dream-RSI.

Rows model discovery-tree transitions and offline policy evaluation. They are
not private facts and must not be treated as proof that online improvement is
safe without a held-out online gate.
"""
import json, shutil, sys
from pathlib import Path

out = Path(sys.argv[1])
if out.exists(): shutil.rmtree(out)
(out / "inputs").mkdir(parents=True)

def make(split):
    rows = []
    for i in range(50):
        rows.append({
            "id": f"{split}:dream-rsi:{i}",
            "task": "replay_policy_evaluation",
            "pod_type": "research",
            "model_track": "Qwen3-GRPO-RSI",
            "question": f"Choose the next exploration action for discovery tree {i+1} using historical replay.",
            "history": {"world_id": f"world:{i%5}", "tree_id": f"tree:{i+1}", "policy_generation": f"g{(i%3)+1}", "visited": ["root", "branch_a"]},
            "evidence": {"candidate_actions": ["expand_a", "expand_b", "prune_low_value", "verify_artifact"], "historical_outcomes": ["novel_solution", "duplicate", "invalid", "timeout"]},
            "target": {"offline_reward": {"novelty": 1.0, "quality": 1.0, "cost": -0.2, "safety": 1.0}, "select": "expand_a", "replay_only": True, "online_validation_required": True},
            "provenance": {"origin_key": f"src:discovery-tree:{i+1}", "world_snapshot": f"world:{i%5}", "dataset_version": "dream_rsi_v1"},
            "assessment": {"offline_policy_value": True, "online_gate": "heldout_world", "anti_leakage": ["no_future_tree", "no_test_outcome_reuse"]}
        })
    return rows

for split in ("train", "dev", "test"):
    (out / "inputs" / f"{split}.json").write_text(json.dumps(make(split), indent=2), encoding="utf-8")
(out / "manifest.json").write_text(json.dumps({"dataset_version": "dream_rsi_v1", "rows_per_split": 50, "track": "Qwen3-GRPO-RSI", "required_gates": ["offline_replay_score", "heldout_world", "online_canary", "revocation_rollback"], "warning": "Replay improvement is off-policy; no automatic promotion to active generation."}, indent=2), encoding="utf-8")
print("created", {s: len(make(s)) for s in ("train", "dev", "test")})
