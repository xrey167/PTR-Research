"""Generation-bound lifecycle tokens for non-commutative neural writes.

This is a small numerical mechanism test, not a replacement for Registry
revocation.  A delete token is transported through writes that happened after
the target write, using conjugation.  The caller still has to validate the
identity, generation and snapshot before applying it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence

import torch

from .registry import InvalidState


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compose(writes: Sequence[torch.Tensor]) -> torch.Tensor:
    if not writes:
        raise ValueError("at least one write is required")
    result = torch.eye(writes[0].shape[0], dtype=writes[0].dtype, device=writes[0].device)
    for write in writes:
        result = result @ write
    return result


@dataclass(frozen=True)
class TransportedLifecycleToken:
    identity_key: str
    generation_key: str
    snapshot_key: str
    write_id: str
    write_digest: str
    matrix: torch.Tensor

    def validate(self, *, identity_key: str, generation_key: str, snapshot_key: str) -> None:
        if (self.identity_key, self.generation_key, self.snapshot_key) != (identity_key, generation_key, snapshot_key):
            raise ValueError("lifecycle token is bound to a different identity, generation or snapshot")


def transported_delete_token(writes: Sequence[tuple[str, torch.Tensor]], target_id: str, *,
                             identity_key: str, generation_key: str, snapshot_key: str) -> TransportedLifecycleToken:
    ids = [write_id for write_id, _ in writes]
    if target_id not in ids:
        raise KeyError(target_id)
    index = ids.index(target_id)
    target = writes[index][1]
    later = [matrix for _, matrix in writes[index + 1:]]
    if later:
        suffix = compose(later)
        matrix = torch.linalg.inv(suffix) @ torch.linalg.inv(target) @ suffix
    else:
        matrix = torch.linalg.inv(target)
    return TransportedLifecycleToken(identity_key, generation_key, snapshot_key, target_id,
                                    _digest(target.detach().cpu().tolist()), matrix)


def apply_token(state: torch.Tensor, token: TransportedLifecycleToken, *,
                identity_key: str, generation_key: str, snapshot_key: str) -> torch.Tensor:
    token.validate(identity_key=identity_key, generation_key=generation_key, snapshot_key=snapshot_key)
    return state @ token.matrix


def validate_registry_binding(token: TransportedLifecycleToken, registry, *, principal: str = "local") -> None:
    """Validate the token's live knowledge generation before neural use.

    Algebraic correctness is insufficient after an edit or revocation.  The
    registry head and its transitive validity are checked at the execution
    boundary; callers can then apply the token to the already validated state.
    """
    try:
        current = registry.head(token.identity_key)
    except Exception as exc:
        raise InvalidState("Lifecycle token references unknown knowledge") from exc
    if current != token.generation_key:
        raise InvalidState("Lifecycle token targets a stale generation")
    registry.snapshot([current], principal)


def static_delete_token(writes: Sequence[tuple[str, torch.Tensor]], target_id: str) -> torch.Tensor:
    for write_id, matrix in writes:
        if write_id == target_id:
            return torch.linalg.inv(matrix)
    raise KeyError(target_id)
