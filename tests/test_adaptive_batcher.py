import threading
import time

import pytest

from neural_pods.adaptive_batcher import AdaptiveBatcher
from neural_pods.adaptive_batcher import BatchedPodHandler
from neural_pods.pod_protocol import PodFanout, PodRequest, PodTransport


def test_batches_compatible_requests_and_preserves_results():
    calls = []
    lock = threading.Lock()

    def run_batch(payloads, key):
        with lock:
            calls.append((key, list(payloads)))
        return [f"{key}:{value * 2}" for value in payloads]

    batcher = AdaptiveBatcher(run_batch, max_batch_size=4, max_wait_ms=10)
    try:
        futures = [batcher.submit(i, key="model-a") for i in range(4)]
        other = batcher.submit(7, key="model-b")
        assert [f.result(timeout=1) for f in futures] == ["model-a:0", "model-a:2", "model-a:4", "model-a:6"]
        assert other.result(timeout=1) == "model-b:14"
        assert any(key == "model-a" and len(payloads) == 4 for key, payloads in calls)
        assert all(key == "model-a" or len(payloads) == 1 for key, payloads in calls)
    finally:
        batcher.close()


def test_expired_request_is_rejected_without_model_call():
    calls = []
    started = threading.Event()

    def run_batch(payloads, key):
        calls.append(payloads)
        started.set()
        time.sleep(0.03)
        return payloads

    batcher = AdaptiveBatcher(run_batch, max_batch_size=1, max_wait_ms=1)
    try:
        first = batcher.submit("first")
        assert started.wait(1)
        future = batcher.submit("late", timeout_ms=1)
        with pytest.raises(TimeoutError):
            future.result(timeout=1)
        assert first.result(timeout=1) == "first"
        assert calls == [["first"]]
        assert batcher.stats()["deadline_drops"] == 1
    finally:
        batcher.close()


def test_batch_errors_propagate_to_every_request():
    def run_batch(payloads, key):
        raise ValueError("inference failed")

    batcher = AdaptiveBatcher(run_batch, max_batch_size=2, max_wait_ms=1)
    try:
        futures = [batcher.submit(i) for i in range(2)]
        for future in futures:
            with pytest.raises(ValueError, match="inference failed"):
                future.result(timeout=1)
        assert batcher.stats()["errors"] == 1
    finally:
        batcher.close()


def test_batched_handler_integrates_with_pod_fanout_and_isolates_principals():
    calls = []

    def run_batch(items, key):
        calls.append((key, items))
        return [{"answer": item[0]["value"] * 2} for item in items]

    handler = BatchedPodHandler(run_batch, max_batch_size=8, max_wait_ms=5)
    transport = PodTransport()
    transport.register("model", "infer", handler)
    try:
        requests = [PodRequest("router", "model", "infer", {"value": i}, principal="alice")
                    for i in range(4)]
        requests += [PodRequest("router", "model", "infer", {"value": 9}, principal="bob")]
        responses = PodFanout(transport, max_workers=5).dispatch(requests)
        assert [response.payload["answer"] for response in responses] == [0, 2, 4, 6, 18]
        assert len(calls) == 2
        assert sorted(len(items) for _, items in calls) == [1, 4]
    finally:
        handler.close()


def test_cancelled_queued_request_is_not_sent_to_model():
    seen = []

    def run_batch(payloads, key):
        seen.extend(payloads)
        return payloads

    batcher = AdaptiveBatcher(run_batch, max_batch_size=4, max_wait_ms=10)
    try:
        first = batcher.submit("keep")
        cancelled = batcher.submit("drop")
        assert cancelled.cancel()
        assert first.result(timeout=1) == "keep"
        time.sleep(0.02)
        assert seen == ["keep"]
        assert batcher.stats()["cancelled"] == 1
    finally:
        batcher.close()


def test_handler_timeout_cancels_queued_work():
    started = threading.Event()
    seen = []

    def run_batch(payloads, key):
        seen.extend(payloads)
        started.set()
        time.sleep(0.05)
        return payloads

    handler = BatchedPodHandler(run_batch, max_batch_size=1, max_wait_ms=20)
    try:
        first = handler._batcher.submit("busy")
        assert started.wait(1)
        request = PodRequest("router", "model", "infer", {"value": "late"}, deadline_ms=2)
        with pytest.raises(TimeoutError):
            handler({"value": "late"}, request)
        assert first.result(timeout=1) == "busy"
        time.sleep(0.03)
        assert seen == ["busy"]
        assert handler.stats()["cancelled"] >= 1
    finally:
        handler.close()
