"""Train a projected Dragonfly router on real Qwen hidden states."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

ENTITIES = ["Müller", "Kern", "Nova", "Atlas", "Rhein", "Elbe"]
TRAIN = ["What is the delivery time for {name}?", "Find the procurement record for {name}.",
         "Which supplier memory contains {name}?", "Summarize the lead time associated with {name}."]
TEST = ["Retrieve transit duration associated with {name}.", "Give me the lead-time fact from {name}."]


class ProjectionRouter(torch.nn.Module):
    def __init__(self, hidden, width, classes):
        super().__init__()
        self.project = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, width), torch.nn.Tanh())
        self.prototypes = torch.nn.Parameter(torch.randn(classes, width) * 0.02)
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        z = torch.nn.functional.normalize(self.project(x), dim=-1)
        p = torch.nn.functional.normalize(self.prototypes, dim=-1)
        return z @ p.T * self.temperature.exp().clamp(max=20)


def run(model_path, device="cuda", dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    backbone = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                                 local_files_only=True).to(device).eval()
    hidden = int(backbone.config.hidden_size)

    def encode(text):
        batch = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = backbone(**batch, output_hidden_states=True, use_cache=False)
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (out.hidden_states[-1] * mask).sum(1) / mask.sum(1).clamp_min(1)
        return pooled[0].float().cpu()

    train_x, train_y, test_x, test_y = [], [], [], []
    for idx, name in enumerate(ENTITIES):
        for template in TRAIN:
            train_x.append(encode(template.format(name=name))); train_y.append(idx)
        for template in TEST:
            test_x.append(encode(template.format(name=name))); test_y.append(idx)
    train_x, train_y = torch.stack(train_x), torch.tensor(train_y)
    test_x, test_y = torch.stack(test_x), torch.tensor(test_y)
    router = ProjectionRouter(hidden, 128, len(ENTITIES))
    opt = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=.01)
    losses = []
    for _ in range(600):
        opt.zero_grad(); logits = router(train_x); loss = torch.nn.functional.cross_entropy(logits, train_y)
        loss.backward(); opt.step(); losses.append(float(loss.detach()))
    with torch.inference_mode():
        logits = router(test_x); probs = logits.softmax(-1); pred = logits.argmax(-1)
    accuracy = float((pred == test_y).float().mean())
    negative = []
    for row, target in zip(probs, test_y):
        negative.extend(float(v) for i, v in enumerate(row) if i != int(target))
    artifact = Path("runs/qwen3b-dragonfly-projected-001.pt")
    torch.save({"state_dict": router.state_dict(), "hidden": hidden, "width": 128,
                "entities": ENTITIES, "train_templates": TRAIN}, artifact)
    saved = torch.load(artifact, map_location="cpu", weights_only=False)
    reloaded = ProjectionRouter(hidden, 128, len(ENTITIES)); reloaded.load_state_dict(saved["state_dict"]); reloaded.eval()
    with torch.inference_mode(): reload_accuracy = float((reloaded(test_x).argmax(-1) == test_y).float().mean())
    return {"model": model_path, "entities": len(ENTITIES), "train_questions": len(train_y),
            "heldout_questions": len(test_y), "top1_accuracy": accuracy,
            "reload_top1_accuracy": reload_accuracy, "artifact": str(artifact),
            "mean_negative_probability": sum(negative) / len(negative),
            "loss_initial": losses[0], "loss_final": losses[-1],
            "scope": "real Qwen hidden states; trained projection/prototype router; synthetic alias-family pilot"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--device", default="cuda")
    args = p.parse_args(); report = run(args.model, args.device)
    Path("runs/qwen3b-dragonfly-projected-001.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
