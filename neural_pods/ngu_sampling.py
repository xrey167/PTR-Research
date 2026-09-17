"""Difficulty-adaptive sampling for search/Pod rollouts.

This is the local, deterministic control-plane analogue of NGU: solved items
are removed after one successful rollout while unsolved items receive another
attempt, up to a bounded budget. It can drive asynchronous workers or GRPO
batch construction without changing lifecycle semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class NGUItem:
    item: T
    attempts: int
    solved: bool


def never_give_up(items: Iterable[T], solve: Callable[[T], bool], *, max_attempts: int = 8) -> list[NGUItem]:
    """Retry hard items and stop spending samples on solved items."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    pending = list(items)
    attempts = {id(item): 0 for item in pending}
    solved: dict[int, bool] = {}
    while pending:
        next_pending: list[T] = []
        for item in pending:
            key = id(item)
            attempts[key] += 1
            ok = bool(solve(item))
            if ok:
                solved[key] = True
            elif attempts[key] < max_attempts:
                next_pending.append(item)
            else:
                solved[key] = False
        pending = next_pending
    return [NGUItem(item, attempts[id(item)], solved[id(item)]) for item in items]
