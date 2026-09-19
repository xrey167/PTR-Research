"""PodExecutor abstraction: runtime-specific executors behind one contract.

The PodExecutor protocol (activate/infer/release/health) lets heterogeneous
runtimes — GPU LoRA serving, gradient-boosted trees, ONNX — live in pods
under the same lifecycle rules: ResourceGovernor leases on activate/release,
deterministic outputs verifiable by the guard, per-pod metrics.

`ExecutorFactory` registers runtime kinds by name; new runtimes are one
registration away. The TreeExecutor demonstrates the CPU path: XGBoost
model, RAM lease, no batching, direct dispatch (Pod-Arm-Design phase P2).
"""
from __future__ import annotations
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ExecutorLease:
    tier: str
    amount_bytes: int


class PodExecutor:
    """Contract every pod runtime implements. Subclasses fill in _infer."""

    runtime: str = "abstract"

    def __init__(self, *, ram_bytes: int = 256 * 1024 * 1024):
        self.ram_bytes = ram_bytes
        self.lease: ExecutorLease | None = None
        self.inferences = 0

    def activate(self, lease: ExecutorLease | None = None) -> None:
        self.lease = lease or ExecutorLease("ram", self.ram_bytes)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.lease is None:
            raise RuntimeError(f"{self.runtime} executor not activated")
        result = self._infer(payload)
        with threading.Lock():
            self.inferences += 1
        return result

    def release(self) -> None:
        self.lease = None

    def health(self) -> bool:
        return self.lease is not None

    def _infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        return {"runtime": self.runtime, "inferences": self.inferences,
                "leased": self.lease is not None}


class TreeExecutor(PodExecutor):
    """XGBoost model pod: deterministic CPU inference from a saved model."""

    runtime = "xgboost"

    def __init__(self, model_path: str | Path, **kwargs):
        super().__init__(**kwargs)
        import xgboost as xgb
        self._model = xgb.Booster()
        self._model.load_model(str(model_path))

    def _infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        features = payload.get("features")
        if features is None:
            raise ValueError("payload requires 'features'")
        matrix = np.array([features], dtype=float)
        probability = float(self._model.predict(
            __import__("xgboost").DMatrix(matrix))[0])
        return {"prediction": 1.0 if probability >= 0.5 else 0.0,
                "probability": round(probability, 6), "deterministic": True}


class ExecutorFactory:
    """Registry: runtimes register under a kind name; pods resolve by kind."""

    def __init__(self):
        self._registry: dict[str, Callable[[], PodExecutor]] = {}
        self._lock = threading.RLock()

    def register(self, kind: str, builder: Callable[[], PodExecutor]) -> None:
        with self._lock:
            if kind in self._registry:
                raise ValueError(f"executor kind already registered: {kind}")
            self._registry[kind] = builder

    def create(self, kind: str) -> PodExecutor:
        with self._lock:
            if kind not in self._registry:
                raise KeyError(f"no executor registered for kind: {kind}")
            return self._registry[kind]()

    def kinds(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._registry))


class GovernedExecutorPool:
    """Activates executors through a ResourceGovernor-compatible budget:
    acquire on activate, release on deactivate, reject on exhaustion."""

    def __init__(self, factory: ExecutorFactory, *, budget_bytes: int):
        self.factory = factory
        self.budget_bytes = budget_bytes
        self.used_bytes = 0
        self.active: dict[str, PodExecutor] = {}
        self._lock = threading.RLock()

    def activate(self, pod_id: str, kind: str) -> PodExecutor:
        executor = self.factory.create(kind)
        with self._lock:
            if self.used_bytes + executor.ram_bytes > self.budget_bytes:
                raise RuntimeError(f"resource budget exhausted for {pod_id}")
            self.used_bytes += executor.ram_bytes
            executor.activate(ExecutorLease("ram", executor.ram_bytes))
            self.active[pod_id] = executor
        return executor

    def release(self, pod_id: str) -> None:
        with self._lock:
            executor = self.active.pop(pod_id, None)
            if executor is not None:
                self.used_bytes -= executor.lease.amount_bytes
                executor.release()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"budget_bytes": self.budget_bytes,
                    "used_bytes": self.used_bytes,
                    "active_pods": sorted(self.active)}


def save_tree_model(model: Any, path: str | Path) -> None:
    Path(path).write_bytes(b"")
    model.save_model(str(path))
