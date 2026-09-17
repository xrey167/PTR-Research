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
