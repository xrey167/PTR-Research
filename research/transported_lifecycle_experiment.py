"""Reproduce the local transported-token mechanism gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_pods.lifecycle_transport import apply_token, compose, static_delete_token, transported_delete_token


def rotation(seed: int, dim: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn((dim, dim), generator=g, dtype=torch.float64)).Q


def run(cases: int = 100) -> dict:
    static_errors, transported_errors = [], []
    for seed in range(cases):
        writes = [(f"w{i}", rotation(seed * 11 + i)) for i in range(6)]
        state = compose([m for _, m in writes])
        expected = compose([m for i, m in writes if i != "w2"])
        static_errors.append(float(torch.linalg.norm(state @ static_delete_token(writes, "w2") - expected)))
        token = transported_delete_token(writes, "w2", identity_key="K", generation_key="G", snapshot_key="S")
        transported_errors.append(float(torch.linalg.norm(apply_token(state, token, identity_key="K", generation_key="G", snapshot_key="S") - expected)))
    return {"cases": cases, "static_median_error": sorted(static_errors)[cases // 2],
            "transported_max_error": max(transported_errors),
            "transported_pass": max(transported_errors) < 1e-12,
            "scope": "float64 orthogonal mechanism test; not pretrained-LLM evidence"}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    Path("runs/transported-lifecycle-local-001.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
