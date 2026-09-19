"""Reflex channel: the main model's native arm to its pods.

The model emits an in-band address signal (an alias it learned during
symlink training). The channel resolves that alias against the currently
authorized generation via the TemporalPortPlane, dispatches to the bound
executor, and passes the result back — with a guarded fallback to an
explicit default pod when the reflex misses. Latency and failover are
counted so the reflex path is measurable against explicit tool-call
routing (Pod-Arm-Design, Phase P1).
"""
from __future__ import annotations
import threading
import time
from typing import Any, Callable

from .symlink import TemporalPortPlane


class ReflexChannel:
    def __init__(self, port_plane: TemporalPortPlane,
                 dispatch: Callable[[str, dict[str, Any]], Any],
                 default_pod: str | None = None,
                 max_latency_s: float = 30.0):
        self.port_plane = port_plane
        self.dispatch = dispatch
        self.default_pod = default_pod
        self.max_latency_s = float(max_latency_s)
        self._lock = threading.RLock()
        self.invocations = 0
        self.reflex_hits = 0
        self.reflex_misses = 0
        self.failovers = 0
        self.errors = 0

    def invoke(self, alias: str, request: dict[str, Any], *, principal: str = "local") -> dict[str, Any]:
        """Resolve the model's address signal and dispatch to the bound pod.

        Returns {"pod", "result", "reflex": bool}. A miss resolves against
        the explicit default pod (deliberate path) instead of failing the
        caller — the arm retracts, the joint stays safe.
        """
        started = time.perf_counter()
        with self._lock:
            self.invocations += 1
        try:
            binding = self.port_plane.resolve(alias, principal=principal)
            if binding["value_handle"] is None:
                raise KeyError("alias bound without a value handle")
            pod_key = str(binding["value_handle"])
        except Exception:
            with self._lock:
                self.reflex_misses += 1
            return self._failover(request, reason="unresolved_alias")
        if time.perf_counter() - started > self.max_latency_s:
            with self._lock:
                self.reflex_misses += 1
            return self._failover(request, reason="resolution_timeout")
        try:
            result = self.dispatch(pod_key, request)
        except Exception:
            with self._lock:
                self.errors += 1
            return self._failover(request, reason="dispatch_error")
        with self._lock:
            self.reflex_hits += 1
        return {"pod": pod_key, "result": result, "reflex": True}

    def _failover(self, request: dict[str, Any], *, reason: str) -> dict[str, Any]:
        if self.default_pod is None:
            with self._lock:
                self.errors += 1
            raise RuntimeError(f"reflex failed and no default pod configured: {reason}")
        with self._lock:
            self.failovers += 1
        result = self.dispatch(self.default_pod, request)
        return {"pod": self.default_pod, "result": result, "reflex": False,
                "failover_reason": reason}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.reflex_hits + self.reflex_misses
            return {"invocations": self.invocations, "reflex_hits": self.reflex_hits,
                    "reflex_misses": self.reflex_misses, "failovers": self.failovers,
                    "errors": self.errors,
                    "hit_rate": self.reflex_hits / total if total else 0.0}
