import pytest
from neural_pods.replication import QuorumReplicator
from neural_pods.fault_injection import FailAction, FailureInjector


def test_quorum_retries_transient_failpoint():
    class Mem:
        def __init__(self): self.rows = {}
        def put(self, k, v): self.rows[k] = v
        def get(self, k): return self.rows.get(k)
    f = FailureInjector({"replica.a.put": FailAction(every=1)})
    # One failed replica is tolerated; the healthy replica reaches quorum.
    r = QuorumReplicator({"a": Mem(), "b": Mem(), "c": Mem()}, quorum=2,
                          injector=f, max_retries=1)
    out = r.write("k", {"v": 1})
    assert out["quorum"] == 2


class Replica:
    def __init__(self): self.data = {}; self.failed = False
    def put(self, key, value):
        if self.failed: raise ConnectionError("offline")
        self.data[key] = dict(value)
    def get(self, key):
        if self.failed: raise ConnectionError("offline")
        return self.data.get(key)


def test_quorum_replication_reads_latest_and_tombstones():
    replicas = {name: Replica() for name in ("a", "b", "c")}
    q = QuorumReplicator(replicas, quorum=2)
    result = q.write("k", {"value": 24})
    assert result["quorum"] == 3 and q.read("k")["payload"]["value"] == 24
    replicas["c"].failed = True
    q.write("k", {"value": 18})
    assert q.read("k")["payload"]["value"] == 18
    q.tombstone("k"); assert q.read("k") is None


def test_quorum_fails_closed_when_majority_is_down():
    replicas = {name: Replica() for name in ("a", "b", "c")}
    q = QuorumReplicator(replicas, quorum=2); replicas["a"].failed = True; replicas["b"].failed = True
    with pytest.raises(RuntimeError): q.write("k", {"value": 1})
