import time
import pytest

from neural_pods.pod_streams import DuplexSession, HypothesisBranch, merge_branches
from neural_pods.registry import InvalidState


def test_verified_branches_merge_by_confidence_and_deduplicate_evidence():
    result = merge_branches([
        HypothesisBranch("b2", "pod:b", "g2", ("k2", "k1"), "second", .7, 20, True),
        HypothesisBranch("b1", "pod:a", "g1", ("k1", "k3"), "first", .9, 40, True),
    ])
    assert result.answer == "first"
    assert result.evidence == ("k1", "k3", "k2")
    assert result.source_branches == ("b1", "b2")


def test_unverified_only_branches_are_rejected():
    with pytest.raises(InvalidState):
        merge_branches([HypothesisBranch("b", "pod:a", "g1", (), None, .5, 1)])


def test_duplex_event_order_interrupt_and_resume():
    session = DuplexSession("duplex:1", capabilities=("resume", "barge_in"))
    first = session.emit("session.created")
    session.commit_turn()
    session.interrupt()
    replay = session.resume(first.sequence)
    assert [e.sequence for e in replay] == [2, 3]
    assert session.epoch == 1
    assert session.close().event_type == "session.closed"


# --- PerceptionStream: regression tests from the 2026-09-20 pod audit ------


def test_events_are_not_counted_as_consumed_without_a_consumer():
    """The drain thread popped events even with no consumer and counted them
    as consumed, so a stream nobody listened to reported perfect delivery —
    which is what the `lossless: true` evidence rested on."""
    from neural_pods.perception import PerceptionStream, synthetic_detector_event

    stream = PerceptionStream(pod_id="no-consumer", max_queue=10)
    try:
        for seq in range(5):
            assert stream.emit(synthetic_detector_event(seq)) == "ok"
        time.sleep(0.05)                       # let the drain thread run
        stats = stream.stats()
        assert stats["consumed"] == 0
        assert stats["queued"] == 5            # backpressure, not a black hole
        assert stats["has_consumer"] is False
    finally:
        stream.close()


def test_a_full_queue_drops_deterministically_without_a_consumer():
    from neural_pods.perception import PerceptionStream, synthetic_detector_event

    stream = PerceptionStream(pod_id="tiny", max_queue=5)
    try:
        outcomes = [stream.emit(synthetic_detector_event(i)) for i in range(50)]
        assert outcomes.count("ok") == 5
        assert outcomes.count("dropped") == 45
        assert stream.stats()["queued"] == 5
    finally:
        stream.close()


def test_a_consumer_drains_the_queue_and_the_count_is_real():
    from neural_pods.perception import PerceptionStream, synthetic_detector_event

    seen = []
    stream = PerceptionStream(pod_id="vision", max_queue=100)
    try:
        stream.set_consumer(seen.append)
        for seq in range(20):
            stream.emit(synthetic_detector_event(seq))
        assert stream.drain(timeout_s=2.0) is True
        assert len(seen) == 20
        assert stream.stats()["consumed"] == 20
    finally:
        stream.close()


def test_the_rate_limit_is_opt_in_and_actually_limits():
    """`max_events_s` used to compute an interval nobody read, next to a
    branch that could never be true: the parameter looked like a guarantee
    and was not one."""
    from neural_pods.perception import PerceptionStream, synthetic_detector_event

    unlimited = PerceptionStream(pod_id="unlimited", max_queue=100)
    limited = PerceptionStream(pod_id="limited", max_queue=100, max_events_s=1.0)
    try:
        assert all(unlimited.emit(synthetic_detector_event(i)) == "ok"
                   for i in range(10))
        outcomes = [limited.emit(synthetic_detector_event(i)) for i in range(10)]
        assert outcomes[0] == "ok"
        assert outcomes[1:] == ["rate_dropped"] * 9
        assert limited.stats()["rate_dropped"] == 9
    finally:
        unlimited.close()
        limited.close()


def test_drain_waits_for_the_consumer_not_just_for_an_empty_queue():
    """The drain loop pops under the lock and calls the consumer after it.

    Between those two points the deque is empty while the event has not been
    delivered, so a `drain()` that waits for an empty deque reports success
    on an undelivered event. With a consumer that does real work — the
    perception benchmark publishes over MQTT — that window is every event.
    """
    import time as time_mod

    from neural_pods.perception import PerceptionStream

    stream = PerceptionStream(pod_id="drain-test")
    seen = []

    def slow(event):
        time_mod.sleep(0.004)
        seen.append(event)

    stream.set_consumer(slow)
    for seq in range(5):
        stream.emit({"seq": seq})
    try:
        assert stream.drain(timeout_s=2.0) is True
        assert len(seen) == 5
        assert stream.stats()["consumed"] == 5
        assert stream.stats()["inflight"] == 0
    finally:
        stream.close()


def test_a_consumer_that_raises_does_not_kill_the_drain_thread():
    """An unguarded callback left the thread dead while `stats()` went on
    reporting `has_consumer: True` and a growing queue forever."""
    from neural_pods.perception import PerceptionStream

    stream = PerceptionStream(pod_id="raise-test")
    seen = []

    def consumer(event):
        if event["seq"] == 0:
            raise RuntimeError("consumer blew up")
        seen.append(event)

    stream.set_consumer(consumer)
    stream.emit({"seq": 0})
    stream.emit({"seq": 1})
    try:
        assert stream.drain(timeout_s=2.0) is True
        stats = stream.stats()
        assert stats["consumer_errors"] == 1
        assert stats["consumed"] == 1
        assert stats["drain_thread_alive"] is True
        assert seen == [{"seq": 1}]
    finally:
        assert stream.close() is True


def test_a_dropped_event_does_not_spend_rate_budget():
    """The rate clock used to advance before the queue-full check, so an
    event the queue refused spent rate budget. The next legitimate emit —
    after the queue had drained — was then refused as `rate_dropped` on
    behalf of an event that was never accepted."""
    import time as time_mod

    from neural_pods.perception import PerceptionStream

    # 50 events/s -> one every 20 ms.
    stream = PerceptionStream(pod_id="rate-test", max_queue=1,
                              max_events_s=50.0)
    try:
        assert stream.emit({"seq": 0}) == "ok"        # fills the queue
        time_mod.sleep(0.025)                         # rate allows the next
        # No consumer yet, so the queue is still full: refused by the QUEUE,
        # and the rate clock must not move for it.
        assert stream.emit({"seq": 1}) == "dropped"

        drained = []
        stream.set_consumer(drained.append)
        assert stream.drain(timeout_s=2.0) is True

        # 25 ms have passed since the last ACCEPTED event, so the rate allows
        # this one. It used to come back "rate_dropped".
        assert stream.emit({"seq": 2}) == "ok"
        stats = stream.stats()
        assert stats["dropped"] == 1
        assert stats["rate_dropped"] == 0
        assert stats["emitted"] == 2
    finally:
        stream.close()


def test_a_perception_run_that_timed_out_is_not_reported_as_completed():
    """`mesh_done.wait(timeout=...)` returns False on a timeout and its
    result was discarded, so a run that delivered half its events still
    recorded `status: completed`. This evidence file has to be re-recorded on
    the server — the defect would have produced the replacement."""
    import ast
    import inspect

    from research import benchmark_perception

    source = inspect.getsource(benchmark_perception.main)
    tree = ast.parse(source.lstrip())

    waits = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "wait"]
    assert waits, "the benchmark no longer waits for delivery"

    # The result must be bound to a name, not dropped on the floor.
    bound = [node for node in ast.walk(tree)
             if isinstance(node, ast.Assign)
             and any(isinstance(v, ast.Call)
                     and getattr(v.func, "attr", "") == "wait"
                     for v in ast.walk(node.value))]
    assert bound, "the wait() result is discarded, so a timeout reads as success"

    # And the status must depend on it rather than being a literal.
    statuses = [node for node in ast.walk(tree)
                if isinstance(node, ast.Dict)
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant) and key.value == "status"]
    assert statuses, "no status field found"
    assert all(not isinstance(value, ast.Constant) for value in statuses), (
        "status is a literal: a partial run cannot be distinguished from a "
        "complete one")
