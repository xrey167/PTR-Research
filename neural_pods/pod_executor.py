"""PodExecutor abstraction: runtime-specific executors behind one contract.

The PodExecutor protocol (activate/infer/release/health) lets heterogeneous
runtimes — GPU LoRA serving, gradient-boosted trees, ONNX — live in pods
under the same lifecycle rules: ResourceGovernor leases on activate/release,
deterministic outputs verifiable by the guard, per-pod metrics.

`ExecutorFactory` registers runtime kinds by name; new runtimes are one
registration away. The TreeExecutor demonstrates the CPU path: XGBoost
model, RAM lease, no batching, direct dispatch (Pod-Arm-Design phase P2).

On leases: the design says the ResourceGovernor docks on here, and
`GovernedExecutorPool` used to reimplement its own byte counter instead.
It now delegates to a real `ResourceGovernor` when one is handed in, and
keeps the standalone counter only as the no-governor fallback — one
admission decision, in one place, whichever path a pod takes.
"""
from __future__ import annotations
import threading
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
        self._lock = threading.RLock()

    def activate(self, lease: ExecutorLease | None = None) -> None:
        self.lease = lease or ExecutorLease("ram", self.ram_bytes)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.lease is None:
            raise RuntimeError(f"{self.runtime} executor not activated")
        result = self._infer(payload)
        # `with threading.Lock():` built a fresh lock per call and therefore
        # guarded nothing; the counter was plain unsynchronised.
        with self._lock:
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
        self._xgb = xgb
        self._model = xgb.Booster()
        self._model.load_model(str(model_path))

    def _infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        features = payload.get("features")
        if features is None:
            raise ValueError("payload requires 'features'")
        matrix = np.array([features], dtype=float)
        probability = float(self._model.predict(self._xgb.DMatrix(matrix))[0])
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
    """Activates executors under a resource budget: acquire on activate,
    release on deactivate, reject on exhaustion.

    Hand in a `ResourceGovernor` and admission goes through it, so executor
    pods and model residency draw on the same budget instead of two
    independent counters that each believe they own the RAM.
    """

    def __init__(self, factory: ExecutorFactory, *, budget_bytes: int | None = None,
                 governor: Any = None):
        if governor is None and budget_bytes is None:
            raise ValueError("need either a governor or a budget_bytes")
        self.factory = factory
        self.governor = governor
        self.budget_bytes = budget_bytes
        self.used_bytes = 0
        self.active: dict[str, PodExecutor] = {}
        self._leases: dict[str, Any] = {}
        self._lock = threading.RLock()

    def activate(self, pod_id: str, kind: str) -> PodExecutor:
        with self._lock:
            # Re-activating a pod id used to overwrite the entry and leak the
            # first executor's bytes forever: the budget shrank with every
            # restart and never came back.
            #
            # The check comes BEFORE factory.create(): a TreeExecutor loads a
            # booster from disk, and a re-activation that is going to be
            # refused should not pay for a model it throws away.
            if pod_id in self.active:
                raise RuntimeError(
                    f"{pod_id} is already active; release it before activating")
            executor = self.factory.create(kind)
            if self.governor is not None:
                lease = self.governor.try_acquire("ram", executor.ram_bytes)
                if lease is None:
                    raise RuntimeError(f"resource budget exhausted for {pod_id}")
                self._leases[pod_id] = lease
            else:
                if self.used_bytes + executor.ram_bytes > self.budget_bytes:
                    raise RuntimeError(f"resource budget exhausted for {pod_id}")
            self.used_bytes += executor.ram_bytes
            executor.activate(ExecutorLease("ram", executor.ram_bytes))
            self.active[pod_id] = executor
        return executor

    def release(self, pod_id: str) -> None:
        with self._lock:
            executor = self.active.pop(pod_id, None)
            if executor is None:
                return
            # Accounted against what was ACQUIRED, not against executor.lease:
            # an executor released directly has lease None and used to make
            # this raise AttributeError, stranding the bytes.
            self.used_bytes -= executor.ram_bytes
            lease = self._leases.pop(pod_id, None)
            if lease is not None and self.governor is not None:
                self.governor.release(lease)
            executor.release()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            # In governed mode the budget lives in the governor, so reporting
            # self.budget_bytes (None) made every caller that computed
            # headroom from it raise TypeError. Ask the owner instead.
            budget = self.budget_bytes
            if budget is None and self.governor is not None:
                budget = getattr(getattr(self.governor, "budget", None),
                                 "ram_bytes", None)
            return {"budget_bytes": budget,
                    "budget_owner": "governor" if self.governor is not None
                                    else "pool",
                    "used_bytes": self.used_bytes,
                    "governed": self.governor is not None,
                    "active_pods": sorted(self.active)}


def save_tree_model(model: Any, path: str | Path) -> None:
    """Persist a booster. (The previous version truncated the file first,
    which did nothing except destroy the old model if save_model then
    failed.)"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
