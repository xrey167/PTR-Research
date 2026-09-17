"""Self-contained demonstration of a lifecycle-bound concept Pod in Qwen."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from neural_pods.continuous_concepts import ContinuousConceptMixer, DecoderConceptInjector


def run():
    torch.manual_seed(20260916)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=41, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32,
    )).eval()
    tokens = torch.tensor([[1, 7, 4, 9]])
    concept = torch.randn(1, tokens.shape[1], 16)
    mixer = ContinuousConceptMixer(16, initial_gate=4.0).eval()
    pod = {
        "artifact_key": "pod:concept:demo",
        "knowledge_key": "knowledge:supplier:muller:x12:lead_time",
        "generation_key": "generation:8",
        "origin_keys": ["src:sap:po_history:92831"],
        "status": "active",
    }
    with torch.inference_mode():
        baseline = model(tokens).logits
        with DecoderConceptInjector(model, mixer, layer_index=0) as injector:
            injector.set_concepts(concept)
            active = model(tokens).logits
        with DecoderConceptInjector(model, mixer, layer_index=0) as injector:
            injector.set_concepts(concept, torch.zeros(1, tokens.shape[1], dtype=torch.bool))
            revoked = model(tokens).logits
    active_delta = float((active - baseline).abs().max())
    revoked_delta = float((revoked - baseline).abs().max())
    result = {
        "pod": pod,
        "input_ids": tokens.tolist(),
        "baseline_top_token": int(baseline[0, -1].argmax()),
        "active_top_token": int(active[0, -1].argmax()),
        "active_max_logit_delta": active_delta,
        "revoked_max_logit_delta": revoked_delta,
        "activation_changed_model": active_delta > 0,
        "revocation_restored_baseline": revoked_delta < 1e-7,
        "scope": "deterministic tiny Qwen demonstration; not language-quality evidence",
    }
    return result


if __name__ == "__main__":
    report = run()
    Path("runs").mkdir(exist_ok=True)
    Path("runs/concept-pod-demo-001.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
