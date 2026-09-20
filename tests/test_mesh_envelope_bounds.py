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
