"""Local J-Space-style state test with lifecycle-bound deletion tokens.

The original private J-Space implementation is unavailable.  This test uses
the checked-in non-commutative operator and Registry contract as the local
compatibility target, and labels the result accordingly.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_pods.lifecycle_transport import compose, transported_delete_token, apply_token, validate_registry_binding
from neural_pods.registry import Registry, InvalidState


def rotation(seed: int, dim: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn((dim, dim), generator=g, dtype=torch.float64)).Q


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "jspace-local-gate-001.json"
    # Use a fresh registry per invocation: revocation is intentionally
    # irreversible, so a previous successful run must not poison a replay.
    reg = Registry(Path(tempfile.mkdtemp(prefix="jspace-gate-", dir=root / ".pytest-tmp")) / "registry.sqlite3")
    try:
        origin = reg.origin("jspace-test", "supplier-x12", "1", {"value": 24}, acl=("buyer",))
        generation = reg.publish("supplier:x12:lead_time", {"value": 24}, [origin], principal="buyer")
        writes = [(f"pod-write-{i}", rotation(100+i)) for i in range(8)]
        state = compose([m for _, m in writes])
        remaining = list(writes)
        errors = []
        for target in ("pod-write-2", "pod-write-6", "pod-write-0"):
            token = transported_delete_token(remaining, target, identity_key="supplier:x12:lead_time",
                                              generation_key=generation, snapshot_key="jspace-s1")
            validate_registry_binding(token, reg, principal="buyer")
            state = apply_token(state, token, identity_key="supplier:x12:lead_time",
                                generation_key=generation, snapshot_key="jspace-s1")
            remaining = [(i, m) for i, m in remaining if i != target]
        expected = compose([m for _, m in remaining])
        errors.append(float(torch.linalg.norm(state-expected)))
        reg.revoke(origin)
        blocked = False
        try:
            validate_registry_binding(token, reg, principal="buyer")
        except InvalidState:
            blocked = True
        result = {"schema":"jspace-local-gate:v1", "passed": errors[0] < 1e-12 and blocked,
                  "writes":8, "deletions":3, "max_state_error":max(errors),
                  "revocation_blocked":blocked,
                  "scope":"local J-Space-style non-commutative operator proxy; original private J-Space not present"}
        out.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    finally:
        reg.close()


if __name__ == "__main__":
    raise SystemExit(main())
