import time
import pytest

from neural_pods.placement import FencedLeader, PlacementDriver


def test_placement_prefers_failure_domains_and_routes_leader():
    pd = PlacementDriver(heartbeat_timeout_s=1)
    pd.register("n1", zone="a", capacity=1)
    pd.register("n2", zone="b", capacity=1)
    pd.register("n3", zone="c", capacity=1)
    placement = pd.assign("research", "r1", replication_factor=3)
    assert len(set(placement.replicas)) == 3
    assert len({pd.metadata()["nodes"][n]["zone"] for n in placement.replicas}) == 3
    assert pd.route("research", "r1", write=True) == placement.leader
    pd.set_health(placement.leader, False)
    with pytest.raises(RuntimeError):
        pd.route("research", "r1", write=True)
    assert pd.route("research", "r1") in placement.replicas[1:]


def test_placement_rebalances_after_node_failure():
    pd = PlacementDriver(heartbeat_timeout_s=1)
    for i, zone in enumerate(("a", "b", "c", "d")):
        pd.register(f"n{i}", zone=zone)
    old = pd.assign("ns", "r", replication_factor=3)
    pd.set_health(old.replicas[0], False)
    new = pd.rebalance("ns", "r")
    assert new.epoch > old.epoch
    assert old.replicas[0] not in new.replicas


def test_reconcile_finds_unhealthy_regions():
    pd = PlacementDriver(heartbeat_timeout_s=1)
    for i, zone in enumerate(("a", "b", "c", "d")):
        pd.register(f"n{i}", zone=zone)
    old = pd.assign("ns", "r", replication_factor=3)
    pd.set_health(old.replicas[1], False)
    changed = pd.reconcile()
    assert len(changed) == 1 and old.replicas[1] not in changed[0].replicas


def test_fenced_leader_rejects_stale_writer():
    leader = FencedLeader(lease_s=0.02)
    first = leader.acquire("n1")
    assert leader.validate("n1", first.fence)
    with pytest.raises(RuntimeError):
        leader.acquire("n2")
    time.sleep(0.03)
    second = leader.acquire("n2")
    assert second.fence > first.fence
    assert not leader.validate("n1", first.fence)
    with pytest.raises(RuntimeError):
        leader.renew("n1", first.fence)
