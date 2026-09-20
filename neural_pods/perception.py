"""Perception stream: a sense-organ pod emits compact events with backpressure.

The detector (in production: ONNX object detection; here a deterministic
synthetic detector) produces bounded, symbolically compressed events. The
stream keeps at most `max_queue` events in flight; an overflowing event is
counted as dropped, never silently queued forever (Pod-Arm-Design, P3).

What this module does NOT do, despite what its docstring used to say: it
does not publish over the mesh. It is an in-process bounded queue with a
drain thread. `research/benchmark_perception.py` is what carries the events
onto a MeshEndpoint, and the mesh numbers in its report come from there.

Rate limiting is opt-in (`max_events_s`). It used to be a constructor
argument that computed an interval nobody read, next to a dead branch that
could never be true — the parameter looked like a guarantee and was not one.

No consumer means backpressure, not a black hole: events stay queued until
one is registered. Previously the drain thread popped them anyway and
counted them as `consumed`, so a stream nobody listened to reported perfect
delivery.

Two further things the queue length alone cannot express, and which the
counters therefore track explicitly:

  * An event that has been popped but whose consumer has not returned yet is
    in flight, not delivered. `drain()` used to wait for an empty deque, so
    it reported success while the last event was still inside the callback —
    for a consumer that does real work (the benchmark publishes over MQTT)
    that turned a "lossless" claim into a race.
  * A consumer that raises used to kill the drain thread silently; `stats()`
    went on reporting `has_consumer: True` while nothing was consumed ever
    again. Consumer errors are counted and the thread survives them.
"""
from __future__ import annotations
import threading
import time
from collections import deque
from typing import Any, Callable


class PerceptionStream:
    def __init__(self, *, pod_id: str, max_queue: int = 1000,
                 max_events_s: float | None = None):
        self.pod_id = pod_id
        self.max_queue = max_queue
        self.max_events_s = max_events_s
        self.min_interval = 1.0 / max_events_s if max_events_s else 0.0
        self.queue: deque[dict[str, Any]] = deque()
        self.dropped = 0
        self.rate_dropped = 0
        self.emitted = 0
        self.consumed = 0
        self.consumer_errors = 0
        self._inflight = 0
        self._last_emit = 0.0
        self._lock = threading.RLock()
        self._consumer: Callable[[dict[str, Any]], None] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                consumer = self._consumer
                event = self.queue.popleft() if (consumer and self.queue) else None
                if event is not None:
                    self._inflight += 1
            if event is None:
                time.sleep(0.002)
                continue
            failed = False
            try:
                consumer(event)
            except Exception:
                failed = True
            with self._lock:
                self._inflight -= 1
                if failed:
                    self.consumer_errors += 1
                else:
                    self.consumed += 1

    def emit(self, event: dict[str, Any]) -> str:
        """Queue one perception event.

        Returns "ok", "dropped" (queue full) or "rate_dropped" (over the
        configured event rate).
        """
        with self._lock:
            now = time.monotonic()
            if self.min_interval and now - self._last_emit < self.min_interval:
                self.rate_dropped += 1
                return "rate_dropped"
            if len(self.queue) >= self.max_queue:
                self.dropped += 1
                return "dropped"
            # The rate clock only advances for an event that was accepted: a
            # dropped event must not spend budget it never got to use.
            self._last_emit = now
            self.queue.append(event)
            self.emitted += 1
            return "ok"

    def set_consumer(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._consumer = callback

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"pod_id": self.pod_id, "emitted": self.emitted,
                    "consumed": self.consumed, "dropped": self.dropped,
                    "rate_dropped": self.rate_dropped,
                    "consumer_errors": self.consumer_errors,
                    "queued": len(self.queue), "inflight": self._inflight,
                    "has_consumer": self._consumer is not None,
                    "drain_thread_alive": self._thread.is_alive()}

    def drain(self, timeout_s: float = 5.0) -> bool:
        """Wait until every queued event has been consumed. False on timeout.

        "Consumed" includes the event the drain thread has already popped but
        whose consumer has not returned yet: an empty deque is not delivery.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self.queue and not self._inflight:
                    return True
            time.sleep(0.002)
        with self._lock:
            return not self.queue and not self._inflight

    def close(self, timeout_s: float = 1.0) -> bool:
        """Stop the drain thread. False if it did not join within the timeout."""
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()


def synthetic_detector_event(seq: int) -> dict[str, Any]:
    """Deterministic compact event (the symbolic compression a real ONNX
    detector would emit: boxes/classes/confidences as small JSON)."""
    x, y = (seq * 37) % 640, (seq * 53) % 480
    return {"seq": seq, "detections": [
        {"class": "component", "box": [x, y, x + 20, y + 20], "conf": 0.91},
        {"class": "pallet", "box": [y, x, y + 40, x + 40], "conf": 0.78},
    ], "ts_ms": int(time.time() * 1000)}
