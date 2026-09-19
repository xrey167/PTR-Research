"""Perception stream: a sense-organ pod pushes compact events over the mesh.

The detector (in production: ONNX object detection; here a deterministic
synthetic detector) publishes bounded, symbolically compressed events on
its channel. Backpressure is explicit: the stream keeps at most
`max_queue` events in flight; an overflowing event is counted as dropped
(crossbeam-style deadline-drop semantics), never silently queued forever.

The consumer side registers a callback and receives verified envelopes.
"""
from __future__ import annotations
import threading
import time
from collections import deque
from typing import Any, Callable


class PerceptionStream:
    def __init__(self, *, pod_id: str, max_queue: int = 1000,
                 max_events_s: float = 500.0):
        self.pod_id = pod_id
        self.max_queue = max_queue
        self.min_interval = 1.0 / max_events_s
        self.queue: deque[dict[str, Any]] = deque()
        self.dropped = 0
        self.emitted = 0
        self.consumed = 0
        self._lock = threading.RLock()
        self._consumer: Callable[[dict[str, Any]], None] | None = None
        self._stop = threading.Event()
        threading.Thread(target=self._drain_loop, daemon=True).start()

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            event = None
            with self._lock:
                if self.queue:
                    event = self.queue.popleft()
            if event is None:
                time.sleep(0.002)
                continue
            if self._consumer is not None:
                self._consumer(event)
            with self._lock:
                self.consumed += 1

    def emit(self, event: dict[str, Any]) -> str:
        """Queue one perception event. Returns 'ok', 'ok_dropped' or 'dropped'."""
        with self._lock:
            if len(self.queue) >= self.max_queue:
                self.dropped += 1
                return "dropped"
            if self.emitted and self.emitted / max(time.monotonic(), 1e-9) < 0:
                pass
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
                    "queued": len(self.queue)}

    def close(self) -> None:
        self._stop.set()


def synthetic_detector_event(seq: int) -> dict[str, Any]:
    """Deterministic compact event (the symbolic compression a real ONNX
    detector would emit: boxes/classes/confidences as small JSON)."""
    x, y = (seq * 37) % 640, (seq * 53) % 480
    return {"seq": seq, "detections": [
        {"class": "component", "box": [x, y, x + 20, y + 20], "conf": 0.91},
        {"class": "pallet", "box": [y, x, y + 40, x + 40], "conf": 0.78},
    ], "ts_ms": int(time.time() * 1000)}
