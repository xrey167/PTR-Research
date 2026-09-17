"""Deadline-aware micro-batching for compatible Pod inference requests.

The batcher is deliberately model/runtime agnostic.  A caller supplies a
``run_batch`` function that receives payloads and returns one result per
payload.  Requests with different compatibility keys never share a batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import Future
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Hashable, Any


@dataclass
class BatchRequest:
    payload: Any
    key: Hashable = "default"
    deadline: float | None = None  # monotonic timestamp
    future: Future = field(default_factory=Future)
    submitted_at: float = field(default_factory=time.monotonic)


class AdaptiveBatcher:
    """Coalesce requests while respecting queue, batch and deadline limits.

    ``max_wait_ms`` bounds queueing delay.  ``target_latency_ms`` is used as a
    conservative deadline guard: if the earliest request cannot afford a full
    inference window, the available requests are dispatched immediately.
    """

    def __init__(
        self,
        run_batch: Callable[[list[Any], Hashable], list[Any]],
        *,
        max_batch_size: int = 64,
        max_wait_ms: float = 2.0,
        target_latency_ms: float = 100.0,
        max_queue: int = 4096,
    ) -> None:
        if max_batch_size < 1 or max_queue < 1:
            raise ValueError("batch and queue limits must be positive")
        self.run_batch = run_batch
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.target_latency_s = target_latency_ms / 1000.0
        self.max_queue = max_queue
        self._queues: dict[Hashable, deque[BatchRequest]] = defaultdict(deque)
        self._condition = threading.Condition()
        self._stopping = False
        self._queued = 0
        self._stats = {"submitted": 0, "completed": 0, "rejected": 0,
                       "batches": 0, "deadline_drops": 0, "cancelled": 0,
                       "errors": 0, "queue_wait_ms_total": 0.0,
                       "queue_wait_ms_max": 0.0, "batch_latency_ms_total": 0.0,
                       "batch_latency_ms_max": 0.0, "batch_items_total": 0}
        self._worker = threading.Thread(target=self._run, name="pod-batcher", daemon=True)
        self._worker.start()

    def submit(self, payload: Any, *, key: Hashable = "default",
               timeout_ms: float | None = None) -> Future:
        now = time.monotonic()
        deadline = None if timeout_ms is None else now + timeout_ms / 1000.0
        request = BatchRequest(payload=payload, key=key, deadline=deadline)
        with self._condition:
            if self._stopping:
                request.future.set_exception(RuntimeError("batcher is stopped"))
                return request.future
            if self._queued >= self.max_queue:
                self._stats["rejected"] += 1
                request.future.set_exception(OverflowError("batch queue is full"))
                return request.future
            self._queues[key].append(request)
            self._queued += 1
            self._stats["submitted"] += 1
            self._condition.notify()
        return request.future

    def close(self, *, cancel_pending: bool = True) -> None:
        with self._condition:
            self._stopping = True
            if cancel_pending:
                for queue in self._queues.values():
                    while queue:
                        request = queue.popleft()
                        self._queued -= 1
                        if not request.future.done():
                            request.future.set_exception(RuntimeError("batcher closed"))
            self._condition.notify_all()
        self._worker.join(timeout=5)

    def stats(self) -> dict[str, int | float]:
        with self._condition:
            batches = self._stats["batches"]
            submitted = self._stats["submitted"]
            return {**self._stats, "queued": self._queued,
                    "active_keys": sum(bool(q) for q in self._queues.values()),
                    "queue_wait_ms_mean": (self._stats["queue_wait_ms_total"] /
                                            max(1, self._stats["batch_items_total"])),
                    "batch_latency_ms_mean": (self._stats["batch_latency_ms_total"] /
                                               max(1, batches)),
                    "batch_size_mean": (self._stats["batch_items_total"] /
                                         max(1, batches))}

    def _select(self) -> list[BatchRequest] | None:
        now = time.monotonic()
        chosen_key = None
        chosen: deque[BatchRequest] | None = None
        # Pick the oldest request across keys for fairness.
        for key, queue in self._queues.items():
            while queue:
                if queue[0].future.cancelled():
                    queue.popleft(); self._queued -= 1; self._stats["cancelled"] += 1
                    continue
                if queue[0].deadline is not None and queue[0].deadline <= now:
                    request = queue.popleft(); self._queued -= 1
                    self._stats["deadline_drops"] += 1
                    if not request.future.done():
                        request.future.set_exception(TimeoutError("request deadline expired in batch queue"))
                    continue
                break
            if queue and (chosen is None or queue[0].submitted_at < chosen[0].submitted_at):
                chosen_key, chosen = key, queue
        if chosen is None:
            return None
        first = chosen[0]
        # If a request is close to its deadline, dispatch what is available.
        remaining = float("inf") if first.deadline is None else first.deadline - now
        if len(chosen) < self.max_batch_size and remaining > self.target_latency_s:
            age = now - first.submitted_at
            if age < self.max_wait_s:
                return None
        items = []
        while chosen and len(items) < self.max_batch_size:
            item = chosen.popleft()
            self._queued -= 1
            if item.future.cancelled():
                self._stats["cancelled"] += 1
                continue
            items.append(item)
        if not chosen:
            self._queues.pop(chosen_key, None)
        return items or None

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stopping and self._queued == 0:
                    self._condition.wait()
                if self._stopping and self._queued == 0:
                    return
                batch = self._select()
                if batch is None:
                    self._condition.wait(timeout=self.max_wait_s)
                    continue
            # A client may cancel after selection but before the worker starts.
            # Drop those payloads before entering model code whenever possible.
            active = []
            with self._condition:
                for item in batch:
                    if item.future.cancelled():
                        self._stats["cancelled"] += 1
                    else:
                        active.append(item)
            if not active:
                continue
            batch = active
            key = batch[0].key
            now = time.monotonic()
            with self._condition:
                waits = [(now - item.submitted_at) * 1000.0 for item in batch]
                self._stats["queue_wait_ms_total"] += sum(waits)
                self._stats["queue_wait_ms_max"] = max(self._stats["queue_wait_ms_max"], max(waits))
            started = time.monotonic()
            try:
                results = self.run_batch([item.payload for item in batch], key)
                if len(results) != len(batch):
                    raise ValueError("run_batch must return one result per request")
                for item, result in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(result)
                        with self._condition: self._stats["completed"] += 1
            except BaseException as exc:
                with self._condition: self._stats["errors"] += 1
                for item in batch:
                    if not item.future.done(): item.future.set_exception(exc)
            finally:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                with self._condition:
                    self._stats["batches"] += 1
                    self._stats["batch_items_total"] += len(batch)
                    self._stats["batch_latency_ms_total"] += elapsed_ms
                    self._stats["batch_latency_ms_max"] = max(
                        self._stats["batch_latency_ms_max"], elapsed_ms)


class BatchedPodHandler:
    """Adapter for registering a batch-capable handler on ``PodTransport``.

    The transport still sees the normal ``handler(payload, request)`` shape;
    compatible calls are coalesced internally.  The default key includes
    lineage and principal fields so model generations and ACL contexts never
    share a model batch accidentally.
    """

    def __init__(self, run_batch: Callable[[list[tuple[Any, Any]], Hashable], list[Any]],
                 *, max_batch_size: int = 64, max_wait_ms: float = 2.0,
                 target_latency_ms: float = 100.0, max_queue: int = 4096,
                 key_fn: Callable[[Any, Any], Hashable] | None = None) -> None:
        self.key_fn = key_fn or self._default_key
        self._batcher = AdaptiveBatcher(self._run, max_batch_size=max_batch_size,
                                        max_wait_ms=max_wait_ms,
                                        target_latency_ms=target_latency_ms,
                                        max_queue=max_queue)
        self._run_batch = run_batch

    @staticmethod
    def _default_key(payload: Any, request: Any) -> Hashable:
        return (getattr(request, "target_generation", None),
                getattr(request, "target_artifact", None),
                getattr(request, "manifest_hash", None),
                getattr(request, "principal", "local"))

    def _run(self, items: list[tuple[Any, Any]], key: Hashable) -> list[Any]:
        return self._run_batch(items, key)

    def __call__(self, payload: Any, request: Any) -> Any:
        timeout_ms = getattr(request, "deadline_ms", None)
        future = self._batcher.submit((payload, request), key=self.key_fn(payload, request),
                                      timeout_ms=timeout_ms)
        try:
            return future.result(timeout=None if timeout_ms is None else max(timeout_ms / 1000.0, 0.001))
        except TimeoutError:
            # Cancel queued work on caller timeout.  If model execution already
            # started, Future.cancel() returns false and the worker finishes
            # safely without affecting the caller.
            future.cancel()
            raise

    def stats(self) -> dict[str, int | float]:
        return self._batcher.stats()

    def close(self) -> None:
        self._batcher.close()
