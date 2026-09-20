"""Reflex channel: the main model's native arm to its pods.

The model emits an in-band address signal (an alias it learned during
symlink training). The channel resolves that alias against the currently
authorized generation via the TemporalPortPlane, dispatches to the bound
executor, and passes the result back — with a guarded fallback to an
explicit default pod when the reflex misses (Pod-Arm-Design, Phase P1).

Counters, and what they mean:

  invocations    every call
  reflex_hits    the alias resolved AND its pod answered
  reflex_misses  everything else — unresolved alias, resolution over the
                 safety timeout, AND a dispatch that raised. A dispatch
                 error used to increment `errors` only, so the invariant the
                 gate checks (`failovers == reflex_misses`) silently assumed
                 no pod ever threw.
  failovers      the default pod served the request instead
  errors         a dispatch raised, on the reflex path or on the default pod

`failovers == reflex_misses` therefore says something now: every miss was
caught by the default pod. `failovers < reflex_misses` means the arm
retracted and the joint did not hold.

Latency is measured, not just claimed: the Pod-Arm design targets under 5 ms
from token to dispatch, and `stats()` reports the resolution and total
percentiles so that target is checkable. `max_latency_s` is a safety timeout
that forces a failover, not the performance target.

Two properties of those samples matter for the number to mean anything:

  * A failed resolution is timed too. It used to return before the sample was
    appended, so `resolve_p95_ms` — the very number the benchmark checks the
    5 ms target against — systematically excluded the slowest resolutions,
    and a channel that missed more looked faster.
  * The samples are bounded (`SAMPLE_WINDOW`). Unbounded lists would grow for
    the lifetime of a serving process and `stats()` sorts them on every call.
    The percentiles therefore describe the most recent window, which `stats()`
    states explicitly rather than implying they cover all time.
"""
from __future__ import annotations
import threading
import time
from collections import Counter, deque
from typing import Any, Callable

from .symlink import TemporalPortPlane

#: How many latency samples the percentiles are computed over.
SAMPLE_WINDOW = 4096


def _percentile(values, fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))], 4)


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
        self.miss_reasons: Counter[str] = Counter()
        self.resolve_ms: deque[float] = deque(maxlen=SAMPLE_WINDOW)
        self.total_ms: deque[float] = deque(maxlen=SAMPLE_WINDOW)

    def invoke(self, alias: str, request: dict[str, Any], *,
               principal: str = "local") -> dict[str, Any]:
        """Resolve the model's address signal and dispatch to the bound pod.

        Returns {"pod", "result", "reflex", "latency_ms"}. A miss resolves
        against the explicit default pod (deliberate path) instead of failing
        the caller — the arm retracts, the joint stays safe.
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
            # A resolution that raised still took time. Sampling it only on
            # the success path would hide exactly the slow cases.
            with self._lock:
                self.resolve_ms.append((time.perf_counter() - started) * 1000)
            return self._miss(started, reason="unresolved_alias", request=request)
        resolve_ms = (time.perf_counter() - started) * 1000
        with self._lock:
            self.resolve_ms.append(resolve_ms)
        if resolve_ms / 1000 > self.max_latency_s:
            return self._miss(started, reason="resolution_timeout", request=request)
        try:
            result = self.dispatch(pod_key, request)
        except Exception:
            with self._lock:
                self.errors += 1
            return self._miss(started, reason="dispatch_error", request=request)
        total_ms = (time.perf_counter() - started) * 1000
        with self._lock:
            self.reflex_hits += 1
            self.total_ms.append(total_ms)
        return {"pod": pod_key, "result": result, "reflex": True,
                "latency_ms": round(total_ms, 4)}

    def _miss(self, started: float, *, reason: str,
              request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.reflex_misses += 1
            self.miss_reasons[reason] += 1
        return self._failover(started, request=request, reason=reason)

    def _failover(self, started: float, *, request: dict[str, Any],
                  reason: str) -> dict[str, Any]:
        if self.default_pod is None:
            with self._lock:
                self.errors += 1
            raise RuntimeError(
                f"reflex failed and no default pod configured: {reason}")
        try:
            result = self.dispatch(self.default_pod, request)
        except Exception as error:
            # The default pod is the joint. If it gives way there is nothing
            # left to retract to, and the caller has to hear about it — but
            # it is counted first, so stats() still describes what happened.
            with self._lock:
                self.errors += 1
                self.miss_reasons["default_pod_failed"] += 1
            raise RuntimeError(
                f"reflex missed ({reason}) and the default pod "
                f"{self.default_pod!r} failed as well: {error!r}") from error
        total_ms = (time.perf_counter() - started) * 1000
        with self._lock:
            self.failovers += 1
            self.total_ms.append(total_ms)
        return {"pod": self.default_pod, "result": result, "reflex": False,
                "failover_reason": reason, "latency_ms": round(total_ms, 4)}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.reflex_hits + self.reflex_misses
            return {
                "invocations": self.invocations,
                "reflex_hits": self.reflex_hits,
                "reflex_misses": self.reflex_misses,
                "failovers": self.failovers,
                "errors": self.errors,
                "hit_rate": self.reflex_hits / total if total else 0.0,
                "miss_reasons": dict(self.miss_reasons),
                # Every miss was caught by the default pod. False means the
                # arm retracted and the joint did not hold.
                "all_misses_covered": self.failovers == self.reflex_misses,
                "resolve_p50_ms": _percentile(self.resolve_ms, 0.5),
                "resolve_p95_ms": _percentile(self.resolve_ms, 0.95),
                "total_p50_ms": _percentile(self.total_ms, 0.5),
                "total_p95_ms": _percentile(self.total_ms, 0.95),
                # The percentiles above cover the last SAMPLE_WINDOW samples,
                # not every invocation since start.
                "latency_samples": len(self.resolve_ms),
                "latency_sample_window": SAMPLE_WINDOW,
            }
