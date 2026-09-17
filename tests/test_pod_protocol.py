import time
import pytest

from neural_pods.pod_protocol import PodRequest, PodTransport, PodFanout
from neural_pods.resource_runtime import RequestTracker


def test_request_preserves_trace_and_rejects_cycles():
    req = PodRequest("router", "research", "search", {}, hop_budget=2)
    nxt = req.next_hop("ranker")
    assert nxt.trace_id == req.trace_id and nxt.hop_budget == 1
    with pytest.raises(ValueError):
        nxt.next_hop("research")


def test_transport_validates_capability_and_deadline():
    transport = PodTransport()
    transport.register("search", "lookup", lambda payload, req: {"count": len(payload)})
    req = PodRequest("router", "search", "lookup", {"q": 1})
    result = transport.dispatch(req)
    assert result.ok and result.payload["count"] == 1
    missing = transport.dispatch(PodRequest("router", "search", "unknown", {}))
    assert missing.error == "capability_unavailable"


def test_transport_reports_deadline_exceeded():
    transport = PodTransport()
    transport.register("slow", "work", lambda payload, req: time.sleep(0.01))
    result = transport.dispatch(PodRequest("router", "slow", "work", {}, deadline_ms=1))
    assert result.error == "deadline_exceeded"


def test_transport_maps_batched_handler_timeout_to_deadline_error():
    transport = PodTransport()
    transport.register("model", "generate", lambda payload, req: (_ for _ in ()).throw(TimeoutError()))
    result = transport.dispatch(PodRequest("router", "model", "generate", {}, deadline_ms=100))
    assert result.error == "deadline_exceeded"


def test_transport_checks_manifest_and_signature_and_retries_transient():
    secret = b"test-secret"; transport = PodTransport(secret=secret, max_retries=2)
    calls = {"n": 0}
    def flaky(payload, request):
        calls["n"] += 1
        if calls["n"] < 3: raise ConnectionError("temporary")
        return {"ok": True}
    transport.register("search", "lookup", flaky, manifest_hash="m1")
    req = PodRequest("router", "search", "lookup", {}, manifest_hash="m1").sign(secret)
    result = transport.dispatch(req)
    assert result.ok and calls["n"] == 3
    bad = PodRequest("router", "search", "lookup", {}, manifest_hash="m2").sign(secret)
    assert transport.dispatch(bad).error == "manifest_mismatch"


def test_fanout_is_parallel_and_preserves_order():
    transport = PodTransport()
    transport.register("a", "work", lambda payload, req: {"pod": "a"})
    transport.register("b", "work", lambda payload, req: {"pod": "b"})
    requests = [PodRequest("router", "a", "work", {}), PodRequest("router", "b", "work", {})]
    responses = PodFanout(transport, max_workers=2).dispatch(requests)
    assert [r.payload["pod"] for r in responses] == ["a", "b"]


def test_transport_records_request_lifecycle_and_payload_metrics():
    tracker = RequestTracker(max_records=8)
    transport = PodTransport(tracker=tracker)
    transport.register("search", "lookup", lambda payload, req: {
        "count": 1, "input_tokens": 3, "output_tokens": 2, "cache_hit": True,
    })
    result = transport.dispatch(PodRequest("router", "search", "lookup", {}))
    assert result.ok
    stats = tracker.stats()
    assert stats["completed"] == 1 and stats["success"] == 1
    assert stats["input_tokens"] == 3 and stats["output_tokens"] == 2
    assert stats["cache_hits"] == 1 and stats["active"] == 0
