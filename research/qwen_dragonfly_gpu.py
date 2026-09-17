"""Evaluate learned Pod addresses from real Qwen hidden states."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
class PodAddress(torch.nn.Module):
    def __init__(self, initial):
        super().__init__(); self.z = torch.nn.Parameter(torch.tensor(initial)); self.bias = torch.nn.Parameter(torch.tensor(0.0))
    def forward(self, queries, features):
        sim = torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(self.z, dim=0)
        return 12 * sim + 2 * features[..., 0] + features[..., 1:].mean(dim=-1) - self.bias

def balanced_loss(logits, labels):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss[labels.bool()].mean() + loss[~labels.bool()].mean()) / 2

ENTITIES = ["Müller", "Kern", "Nova", "Atlas", "Rhein", "Elbe"]
TRAIN = ["What is the delivery time for {name}?", "Find the procurement record for {name}.",
         "Which supplier memory contains {name}?"]
TEST = ["Retrieve transit duration associated with {name}.", "Give me the lead-time fact from {name}."]


def run(model_path, device="cuda", dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                             local_files_only=True).to(device).eval()

    def encode(text):
        batch = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1][0, -1].float().cpu()

    vectors = {}
    for name in ENTITIES:
        vectors[name] = {q: encode(q.format(name=name)) for q in TRAIN + TEST}
    models = {}
    for name in ENTITIES:
        positives = [vectors[name][q] for q in TRAIN]
        negatives = [vectors[other][q] for other in ENTITIES if other != name for q in TRAIN]
        x = torch.stack(positives + negatives)
        y = torch.tensor([1.] * len(positives) + [0.] * len(negatives))
        model_address = PodAddress(positives[0].tolist())
        features = torch.ones((len(x), 9)); features[:, 0] = x @ positives[0]
        opt = torch.optim.AdamW(model_address.parameters(), lr=.05)
        for _ in range(160):
            opt.zero_grad(); loss = balanced_loss(model_address(x, features), y); loss.backward(); opt.step()
        models[name] = model_address.eval()
    correct, negative = [], []
    for name in ENTITIES:
        for template in TEST:
            v = vectors[name][template]
            scores = {}
            for other in ENTITIES:
                f = torch.ones(9); f[0] = v @ models[other].z.detach()
                with torch.inference_mode(): scores[other] = float(models[other](v, f).sigmoid())
            ranked = sorted(scores, key=scores.get, reverse=True)
            correct.append(float(ranked[0] == name))
            negative.extend(scores[o] for o in scores if o != name)
    return {"model": model_path, "entities": len(ENTITIES), "heldout_questions": len(correct),
            "top1_accuracy": sum(correct) / len(correct),
            "negative_mean_score": sum(negative) / len(negative),
            "scope": "real Qwen hidden states; synthetic alias-family pilot"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--device", default="cuda")
    args = p.parse_args(); report = run(args.model, args.device)
    Path("runs/qwen3b-dragonfly-pilot-001.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
