"""Multi-Pod demonstration: context, math and model concepts over tiny Qwen."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from neural_pods.continuous_concepts import ContinuousConceptMixer, DecoderConceptInjector


PODS = {
    "context": {"artifact_key": "pod:context:muller", "knowledge_key": "supplier:muller:x12:lead_time",
                "prompt_prefixes": ("who", "which supplier", "where", "what context")},
    "math": {"artifact_key": "pod:math:lead-time", "knowledge_key": "math:lead_time_days",
             "prompt_prefixes": ("how many", "calculate", "what is", "days")},
    "model": {"artifact_key": "pod:model:policy", "knowledge_key": "policy:approval_threshold",
              "prompt_prefixes": ("which policy", "should", "what rule", "approval")},
}


def classify(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ("calculate", "how many", "days")):
        return "math"
    if any(x in q for x in ("policy", "rule", "approval", "should")):
        return "model"
    return "context"


def run():
    torch.manual_seed(20260916)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=41, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32,
    )).eval()
    mixers = {name: ContinuousConceptMixer(16, initial_gate=4.0).eval() for name in PODS}
    concepts = {name: torch.randn(1, 4, 16) for name in PODS}
    questions = [
        "Which supplier is associated with X12?",
        "How many days is the lead time?",
        "Which policy requires approval?",
        "What context is linked to Müller?",
        "Calculate the delivery buffer in days.",
        "Should approval be requested?",
    ]
    rows = []
    for index, question in enumerate(questions):
        ids = torch.tensor([[1, 7 + index % 5, 4, 9]])
        selected = classify(question)
        with torch.inference_mode():
            baseline = model(ids).logits
            with DecoderConceptInjector(model, mixers[selected], 0) as injector:
                injector.set_concepts(concepts[selected])
                active = model(ids).logits
            wrong = next(name for name in PODS if name != selected)
            with DecoderConceptInjector(model, mixers[wrong], 0) as injector:
                injector.set_concepts(concepts[wrong])
                wrong_logits = model(ids).logits
            with DecoderConceptInjector(model, mixers[selected], 0) as injector:
                injector.set_concepts(concepts[selected], torch.zeros(1, 4, dtype=torch.bool))
                revoked = model(ids).logits
        rows.append({
            "question": question,
            "selected_pod_type": selected,
            "selected_artifact": PODS[selected]["artifact_key"],
            "active_delta": float((active - baseline).abs().max()),
            "wrong_pod_delta": float((wrong_logits - baseline).abs().max()),
            "revoked_delta": float((revoked - baseline).abs().max()),
            "revocation_restores_baseline": bool(torch.equal(revoked, baseline)),
        })
    return {"pod_types": list(PODS), "questions": rows,
            "scope": "routing/injection demonstration; not language-quality evidence"}


if __name__ == "__main__":
    report = run()
    Path("runs").mkdir(exist_ok=True)
    Path("runs/pod-gallery-demo-001.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
