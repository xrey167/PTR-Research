"""Hop budget and deadline: the bounds that now travel with the message.

`PodLink` has carried `hop_budget` and `deadline_ms` since the contract was
written and `PodLink.validate()` enforces both — but nothing of either
reached the wire. The moment a request crossed the mesh, the receiving pod
built a fresh link with a fresh budget, so a ring of pods could pass work
around forever while every individual hop validated.

These tests exercise `envelope_refusal`, which both the arriving side
(`_on_message`) and the departing side (`forward`) defer to. It is module
level and pure precisely so the rule can be tested without a broker: a rule
that only runs against live MQTT is a rule nobody runs.
"""
import time

import pytest

from neural_pods.mesh import envelope_refusal


def _envelope(**fields):
    base = {"body": {"q": 1}, "ts_ms": time.time() * 1000}
    base.update(fields)
    return base


def test_an_unbounded_envelope_arrives_but_cannot_be_forwarded():
    """Every message the system sent before the bounds existed is unbounded.
    Accepting those keeps working; passing one further is what makes a loop."""
    envelope = _envelope()
    assert envelope_refusal(envelope, pod_id="pod-a") is None
    assert envelope_refusal(envelope, pod_id="pod-a", forwarding=True) == "unbounded"


def test_a_budget_of_one_is_the_last_hop():
    envelope = _envelope(hop_budget=1, visited=[])
    # It may be acted on here...
    assert envelope_refusal(envelope, pod_id="pod-b") is None
    # ...but not passed on: the next pod would have nothing left.
    assert envelope_refusal(envelope, pod_id="pod-b", forwarding=True) == "hop_exhausted"


def test_a_spent_budget_is_refused_on_arrival_too():
    """Enforced before any handler sees it. A handler that answers is one
    more hop, so accepting an out-of-budget message makes the bound
    advisory."""
    envelope = _envelope(hop_budget=0, visited=["pod-a"])
    assert envelope_refusal(envelope, pod_id="pod-b") == "hop_exhausted"


def test_a_pod_refuses_to_forward_a_message_it_has_already_seen():
    envelope = _envelope(hop_budget=5, visited=["pod-a", "pod-b"])
    assert envelope_refusal(envelope, pod_id="pod-b", forwarding=True) == "cycle"
    assert envelope_refusal(envelope, pod_id="pod-c", forwarding=True) is None


def test_the_deadline_counts_from_the_original_publish_not_the_last_hop():
    """The original ts_ms is what makes the deadline mean anything:
    re-stamping it per hop gives every hop the full deadline again."""
    started = (time.time() - 2.0) * 1000          # two seconds ago
    envelope = _envelope(ts_ms=started, deadline_ms=1500, hop_budget=5)
    assert envelope_refusal(envelope, pod_id="pod-b") == "expired"
    assert envelope_refusal(envelope, pod_id="pod-b", forwarding=True) == "expired"

    fresh = _envelope(ts_ms=time.time() * 1000, deadline_ms=1500, hop_budget=5)
    assert envelope_refusal(fresh, pod_id="pod-b") is None


def test_an_expired_envelope_is_refused_even_with_budget_left():
    envelope = _envelope(ts_ms=(time.time() - 10) * 1000, deadline_ms=100,
                         hop_budget=99)
    assert envelope_refusal(envelope, pod_id="pod-b") == "expired"


def test_a_ring_of_pods_terminates():
    """The property the whole mechanism exists for, played out: three pods
    hand the message on in a circle. Without the budget on the wire this
    never stops."""
    pods = ["pod-a", "pod-b", "pod-c"]
    envelope = _envelope(hop_budget=3, visited=[])
    hops = 0
    index = 0
    while hops < 100:
        pod = pods[index % len(pods)]
        refusal = envelope_refusal(envelope, pod_id=pod, forwarding=True)
        if refusal is not None:
            break
        envelope = dict(envelope,
                        hop_budget=envelope["hop_budget"] - 1,
                        visited=[*envelope["visited"], pod])
        hops += 1
        index += 1
    assert hops == 2
    assert refusal == "hop_exhausted"


def test_a_pod_link_can_be_published_under_its_own_bounds():
    """The link contract is where the numbers come from; publish(link=...)
    is what puts them on the wire."""
    from neural_pods.pod_contract import PodLink

    link = PodLink(trace_id="t-9", source_pod_id="pod-a", target_pod_id="pod-b",
                   target_generation="g", artifact_id="a", transport="mqtt",
                   capability="c", acl=("*",), deadline_ms=1500, hop_budget=3)
    envelope = _envelope(hop_budget=link.hop_budget,
                         deadline_ms=link.deadline_ms,
                         trace_id=link.trace_id, visited=list(link.visited))
    assert envelope_refusal(envelope, pod_id="pod-b") is None
    assert envelope["trace_id"] == "t-9"
    assert envelope["hop_budget"] == 3


# ---------------------------------------------------------------------------
# Malformed bounds. These three fields come off the wire, so a sender chooses
# their types — and `envelope_refusal` does int()/float() on them. Raising
# reached the paho callback, which re-raises into its network loop
# (suppress_exceptions defaults to False in 2.1.0) and can end the receiving
# thread. A field added to BOUND a request would have become a way to silence
# the pod receiving it, and the message would not even have been counted.


@pytest.mark.parametrize("envelope", [
    {"hop_budget": "x"},
    {"hop_budget": "3; DROP"},
    {"hop_budget": []},
    {"hop_budget": {"n": 3}},
    {"hop_budget": None, "deadline_ms": "soon", "ts_ms": 1.0},
    {"hop_budget": 3, "visited": "pod-a"},          # a string, not a list
    {"hop_budget": 3, "visited": 7},
])
def test_a_malformed_bound_is_refused_not_raised(envelope):
    base = _envelope(**envelope)
    assert envelope_refusal(base, pod_id="pod-a") == "malformed"
    assert envelope_refusal(base, pod_id="pod-a", forwarding=True) == "malformed"


def test_a_malformed_deadline_is_refused():
    envelope = _envelope(deadline_ms="eventually", ts_ms=time.time() * 1000)
    assert envelope_refusal(envelope, pod_id="pod-a") == "malformed"


def test_a_numeric_string_budget_is_accepted():
    """JSON round-trips through other languages; "3" is well-formed input,
    not an attack. Refusing it would break interoperability for no gain."""
    envelope = _envelope(hop_budget="3", visited=[])
    assert envelope_refusal(envelope, pod_id="pod-a") is None


def test_a_deadline_without_a_timestamp_leaves_the_deadline_unenforced():
    """ts_ms is stamped by _publish_raw, so a deadline without one comes from
    an older sender. That is not malformed — it just cannot be checked."""
    envelope = {"body": {}, "deadline_ms": 1500, "hop_budget": 3, "visited": []}
    assert envelope_refusal(envelope, pod_id="pod-a") is None


def test_the_bounds_are_evaluated_inside_the_receive_guard():
    """The structural half of the fix: whatever envelope_refusal does, the
    call site must sit where an exception is counted rather than escaping
    into the paho network loop."""
    import ast
    import inspect

    from neural_pods import mesh

    source = inspect.getsource(mesh.MeshEndpoint._on_message)
    tree = ast.parse(source.lstrip())
    handler = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_on_message")
    guarded = [n for n in ast.walk(handler) if isinstance(n, ast.Try)]
    assert guarded, "_on_message has no guard at all"

    def calls_refusal(node):
        return any(isinstance(inner, ast.Call)
                   and getattr(inner.func, "id", "") == "envelope_refusal"
                   for inner in ast.walk(node))

    assert any(calls_refusal(block) for block in guarded), (
        "envelope_refusal() is called outside the try block: a malformed "
        "field would escape into the paho network loop")
