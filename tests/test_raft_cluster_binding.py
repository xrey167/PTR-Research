import pytest

raft = pytest.importorskip("neural_pods_raft")


def test_three_node_binding_elects_and_replicates():
    nodes = {i: raft.RaftNode(i, [1, 2, 3]) for i in (1, 2, 3)}
    queue, applied = [], {i: [] for i in nodes}

    def emit(src):
        for target, msg in zip(nodes[src].pending_message_targets(), nodes[src].pending_messages()):
            queue.append((target, msg))
        applied[src].extend(nodes[src].ack_ready())

    nodes[1].campaign_pending(); emit(1)
    while queue:
        target, msg = queue.pop(0)
        if nodes[target].step_message(msg) is not None:
            emit(target)
    assert "Leader" in nodes[1].status()[2]
    nodes[1].propose_pending(b"cluster-proof"); emit(1)
    while queue:
        target, msg = queue.pop(0)
        if nodes[target].step_message(msg) is not None:
            emit(target)
    assert all(b"cluster-proof" in applied[i] for i in nodes)
