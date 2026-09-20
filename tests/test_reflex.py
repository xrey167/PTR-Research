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


# --- regression tests for the P1 defects found in the pod audit ------------


def test_a_dispatch_error_counts_as_a_miss_not_only_as_an_error():
    """`failovers == reflex_misses` is what the gate checks. A raising pod
    used to bump `failovers` without bumping `reflex_misses`, so the check
    silently assumed no pod ever threw."""
    channel, plane, _calls = _channel()
    plane.bind("adapter:broken", ["broken"], value_handle="pod-broken")

    def failing(pod_key, request):
        if pod_key == "pod-broken":
            raise RuntimeError("engine down")
        return "ok"

    channel.dispatch = failing
    channel.invoke("broken", {})

    stats = channel.stats()
    assert stats["reflex_misses"] == 1 and stats["failovers"] == 1
    assert stats["errors"] == 1
    assert stats["all_misses_covered"] is True
    assert stats["miss_reasons"] == {"dispatch_error": 1}


def test_a_failing_default_pod_is_reported_not_raised_raw():
    """The default pod is the joint. If it gives way the caller has to hear
    about it — the old code let the raw exception escape uncounted."""
    channel, _plane, _calls = _channel()

    def everything_down(pod_key, request):
        raise RuntimeError("everything is down")

    channel.dispatch = everything_down
    try:
        channel.invoke("totally-unknown-alias", {})
        raise AssertionError("expected the failing default pod to be reported")
    except RuntimeError as error:
        assert "default pod" in str(error) and "failed as well" in str(error)

    stats = channel.stats()
    assert stats["reflex_misses"] == 1
    assert stats["failovers"] == 0
    assert stats["all_misses_covered"] is False       # the joint did not hold
    assert stats["miss_reasons"]["default_pod_failed"] == 1


def test_latency_is_measured_not_only_claimed():
    """Pod-Arm-Design targets under 5 ms from token to dispatch. That was
    unmeasurable: the channel recorded no latency at all."""
    channel, plane, _calls = _channel()
    plane.bind("adapter:reader-gen5", ["reader-gen5"], value_handle="pod-gen5")
    for _ in range(20):
        assert channel.invoke("reader-gen5", {"q": "hi"})["latency_ms"] >= 0

    stats = channel.stats()
    assert stats["resolve_p50_ms"] is not None
    assert stats["total_p50_ms"] is not None
    assert stats["total_p95_ms"] >= stats["total_p50_ms"]
    assert stats["reflex_hits"] == 20 and stats["hit_rate"] == 1.0


def test_miss_reasons_separate_the_ways_a_reflex_fails():
    channel, plane, _calls = _channel()
    plane.bind("adapter:broken", ["broken"], value_handle="pod-broken")

    def failing(pod_key, request):
        if pod_key == "pod-broken":
            raise RuntimeError("engine down")
        return "ok"

    channel.dispatch = failing
    channel.invoke("broken", {})                # dispatch_error
    channel.invoke("totally-unknown-alias", {})  # unresolved_alias

    assert channel.stats()["miss_reasons"] == {"dispatch_error": 1,
                                               "unresolved_alias": 1}
    assert channel.stats()["all_misses_covered"] is True


def test_a_failed_resolution_is_timed_too():
    """`resolve_ms` used to be appended only on the success path, so
    `resolve_p95_ms` — the number the Pod-Arm 5 ms target is checked against —
    systematically excluded the slowest resolutions. A channel that missed
    more looked faster."""
    channel, plane, _calls = _channel()
    plane.bind("adapter:reader-gen5", ["reader-gen5"], value_handle="pod-gen5")

    channel.invoke("reader-gen5", {"q": "hit"})
    channel.invoke("no-such-alias", {"q": "miss"})

    stats = channel.stats()
    assert stats["reflex_hits"] == 1
    assert stats["reflex_misses"] == 1
    # One sample per invocation, misses included.
    assert stats["latency_samples"] == 2
    assert stats["resolve_p95_ms"] is not None


def test_the_latency_samples_are_bounded():
    """Unbounded lists grow for the lifetime of a serving process, and
    `stats()` sorts them on every call."""
    from neural_pods.reflex import SAMPLE_WINDOW

    channel, plane, _calls = _channel()
    plane.bind("adapter:reader-gen5", ["reader-gen5"], value_handle="pod-gen5")
    for _ in range(SAMPLE_WINDOW + 25):
        channel.invoke("reader-gen5", {"q": "x"})

    stats = channel.stats()
    assert stats["invocations"] == SAMPLE_WINDOW + 25
    assert stats["latency_samples"] == SAMPLE_WINDOW
    assert stats["latency_sample_window"] == SAMPLE_WINDOW
    assert len(channel.total_ms) == SAMPLE_WINDOW
