"""Deterministic failpoints for control-plane reliability tests.

This mirrors the useful part of TiKV's fail-rs: failures are opt-in, named,
and disabled in normal builds.  Counts are deterministic so a test can inject
the second WAL or transport operation without relying on timing.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Mapping


class InjectedFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class FailAction:
    kind: str = "error"
    delay_s: float = 0.0
    every: int = 1


class FailureInjector:
    def __init__(self, actions: Mapping[str, FailAction] | None = None):
        self._actions = dict(actions or {})
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def hit(self, name: str) -> None:
        action = self._actions.get(name)
        if action is None:
            return
        with self._lock:
            count = self._counts.get(name, 0) + 1
            self._counts[name] = count
        if action.every > 1 and count % action.every:
            return
        if action.delay_s:
            time.sleep(action.delay_s)
        if action.kind in {"error", "raise", "panic"}:
            raise InjectedFailure(f"failpoint:{name}:count={count}")

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

