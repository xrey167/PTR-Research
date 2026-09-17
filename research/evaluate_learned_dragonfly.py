"""Small learned Pod-address pilot with held-out question formulations."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from neural_pods.dragonfly import PodAddress, balanced_loss


ENTITIES = [("muller", "Müller"), ("kern", "Kern"), ("nova", "Nova"),
            ("atlas", "Atlas"), ("rhein", "Rhein"), ("elbe", "Elbe")]


class Encoder:
    def __init__(self):
        self.words = {name: torch.nn.functional.normalize(torch.randn(16, generator=torch.Generator().manual_seed(i)), dim=0)
                      for i, (name, _) in enumerate(ENTITIES)}

    def encode(self, text, normalize_embeddings=True):
        lower = text.casefold()
        vector = torch.zeros(16)
        for name, label in ENTITIES:
            if name in lower or label.casefold() in lower:
                vector += self.words[name]
        if vector.norm() == 0:
            vector[0] = 1
        return torch.nn.functional.normalize(vector, dim=0) if normalize_embeddings else vector


def run():
    encoder = Encoder(); models = {}
    train_phrases = ["{label} delivery time for component X12", "find {label} procurement record",
                     "which Pod contains {label} lead time?"]
    test_phrases = ["tell me the transit duration from {label} for X12",
                    "retrieve the supplier memory associated with {label}"]
    for name, label in ENTITIES:
        positive = [p.format(label=label) for p in train_phrases]
        negative = [template.format(label=other) for template in train_phrases
                    for _, other in ENTITIES if other != label]
        model = PodAddress(encoder.encode(label).tolist())
        x = torch.stack([encoder.encode(q) for q in positive + negative])
        y = torch.tensor([1.] * len(positive) + [0.] * len(negative))
        features = torch.ones((len(x), 9)); features[:, 0] = x @ encoder.encode(label)
        opt = torch.optim.AdamW(model.parameters(), lr=.05)
        for _ in range(120):
            opt.zero_grad(); loss = balanced_loss(model(x, features), y); loss.backward(); opt.step()
        models[name] = model.eval()
    correct, negative_scores = [], []
    for name, label in ENTITIES:
        for template in test_phrases:
            q = template.format(label=label); v = encoder.encode(q)
            scores = {}
            for other, other_label in ENTITIES:
                f = torch.ones(9); f[0] = v @ encoder.encode(other_label)
                with torch.inference_mode():
                    scores[other] = float(models[other](v, f).sigmoid())
            ranked = sorted(scores, key=scores.get, reverse=True)
            correct.append(float(ranked[0] == name))
            negative_scores.extend(scores[o] for o in scores if o != name)
    return {"entities": len(ENTITIES), "heldout_questions": len(correct),
            "top1_accuracy": sum(correct) / len(correct),
            "negative_mean_score": sum(negative_scores) / len(negative_scores),
            "scope": "learned alias-family pilot; synthetic encoder and held-out formulations"}


if __name__ == "__main__":
    result = run()
    Path("runs/learned-dragonfly-pilot-001.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
