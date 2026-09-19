from neural_pods.registry import Registry
from neural_pods.symlink import TemporalPortPlane
from neural_pods.reflex import ReflexChannel


def _publish(registry, knowledge_key):
    origin = registry.origin("reflex_tests", knowledge_key, 1, {"kind": "test"})
    return registry.publish(knowledge_key, {"pod_type": "lora"}, parents=[origin])


def _channel(default_pod="reader-gen5"):
    registry = Registry(":memory:")
    plane = TemporalPortPlane(registry)
    _publish(registry, "adapter:reader-gen5")
    _publish(registry, "adapter:broken")
    calls = []

    def dispatch(pod_key, request):
        calls.append((pod_key, request))
        return {"answer": f"from-{pod_key}", "echo": request.get("q")}

    return ReflexChannel(plane, dispatch, default_pod=default_pod), plane, calls


def test_reflex_resolves_bound_alias_and_dispatches():
    channel, plane, calls = _channel()
    plane.bind("adapter:reader-gen5", ["reader-gen5"], value_handle="pod-gen5")
    result = channel.invoke("reader-gen5", {"q": "hi"})
    assert result["reflex"] is True
    assert result["pod"] == "pod-gen5"
    assert result["result"]["echo"] == "hi"
    assert channel.stats()["reflex_hits"] == 1
    assert calls == [("pod-gen5", {"q": "hi"})]


def test_reflex_misses_fall_back_to_default_pod():
    channel, plane, calls = _channel()
    result = channel.invoke("totally-unknown-alias", {"q": "x"})
    assert result["reflex"] is False
    assert result["pod"] == "reader-gen5"
    assert result["failover_reason"] == "unresolved_alias"
    assert channel.stats()["failovers"] == 1
    assert channel.stats()["reflex_misses"] == 1


def test_dispatch_error_also_retracts_to_default():
    channel, plane, calls = _channel()
    plane.bind("adapter:broken", ["broken"], value_handle="pod-broken")

    def failing(pod_key, request):
        if pod_key == "pod-broken":
            raise RuntimeError("engine down")
        return "ok"

    channel.dispatch = failing
    result = channel.invoke("broken", {})
    assert result["reflex"] is False
    assert result["pod"] == "reader-gen5"
    assert channel.stats()["errors"] == 1
    assert channel.stats()["failovers"] == 1


def test_reflex_without_default_pod_raises():
    channel, plane, _ = _channel(default_pod=None)
    try:
        channel.invoke("unknown", {})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as error:
        assert "no default pod" in str(error)


def test_stats_hit_rate():
    channel, plane, _ = _channel()
    plane.bind("adapter:reader-gen5", ["a"], value_handle="pod-a")
    channel.invoke("a", {})
    channel.invoke("nope", {})
    stats = channel.stats()
    assert stats["invocations"] == 2 and stats["hit_rate"] == 0.5
