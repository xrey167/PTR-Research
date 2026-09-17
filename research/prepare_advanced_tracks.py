"""Create small, isolated tracks for SID-style retrieval, multiplex reasoning,
and realtime duplex runtime evaluation. These are contracts, not fake private
knowledge and not a claim that the model already implements the algorithms.
"""
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
if out.exists():
    import shutil; shutil.rmtree(out)
(out / "inputs").mkdir(parents=True)

def rows(split):
    result = []
    for i in range(50):
        result.append({"id": f"{split}:advanced:sid_retrieval:{i}", "task": "sid_retrieval", "model_track": "Qwen3-GRPO-Retrieval", "pod_type": "research", "question": f"Find and rank the necessary documents for retrieval case {i+1}.", "evidence": {"corpus": f"namespace:research:{i%5}", "gold_documents": [f"doc:{i}:a", f"doc:{i}:b", f"doc:{i}:c"]}, "target": {"ranked_documents": [f"doc:{i}:a", f"doc:{i}:b", f"doc:{i}:c"], "metric": "NDCG", "stop": "all_required_documents_found", "allow_overretrieve": True}, "assessment": {"retrieval_reward": ["ndcg", "latency", "tool_efficiency", "format"], "document_centric": True}})
        result.append({"id": f"{split}:advanced:multiplex_reasoning:{i}", "task": "multiplex_reasoning", "model_track": "Qwen3-GRPO-Multiplex", "pod_type": "reasoning", "question": f"Solve reasoning case {i+1} with confidence-aware branch and merge.", "evidence": {"domain": "math_or_symbolic", "branch_count": 4 + i % 4}, "target": {"branches": 4 + i % 4, "merge": "confidence_weighted", "emit": "final_answer_only", "reward": ["correctness", "shorter_trajectory", "calibrated_confidence"]}, "assessment": {"tokenwise_branch_merge": True, "experimental": True}})
        result.append({"id": f"{split}:advanced:duplex_runtime:{i}", "task": "duplex_runtime", "model_track": "vLLM-Omni-Duplex", "pod_type": "runtime", "question": f"Handle realtime session event sequence {i+1}.", "evidence": {"events": ["session.created", "audio.append", "response.speak", "barge_in", "session.resume"]}, "target": {"session_mode": "duplex", "preserve": ["session_id", "server_event_seq", "epoch"], "measure": ["first_text_ms", "first_audio_ms", "audio_rtf"], "fallback": "turn_based"}, "assessment": {"runtime_only": True, "no_model_knowledge_claim": True}})
    return result

for split in ("train", "dev", "test"):
    (out / "inputs" / f"{split}.json").write_text(json.dumps(rows(split), ensure_ascii=False, indent=2), encoding="utf-8")
(out / "manifest.json").write_text(json.dumps({"dataset_version": "advanced_v1", "rows_per_split": 150, "tracks": {"Qwen3-GRPO-Retrieval": "SID-style document ranking with NDCG", "Qwen3-GRPO-Multiplex": "experimental branch-and-merge reasoning", "vLLM-Omni-Duplex": "runtime protocol/evaluation only"}, "warning": "Do not train duplex runtime rows as language knowledge; do not claim multiplex implementation until algorithm integration is present."}, indent=2), encoding="utf-8")
print("created", {s: len(rows(s)) for s in ("train", "dev", "test")})
