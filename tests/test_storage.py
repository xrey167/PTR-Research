import pytest

from neural_pods.storage import PodStorage


class FakeRedis:
    """Dict-backed redis stub: set/get/ex with TTL ignored."""
    def __init__(self):
        self.data = {}
        self.set_calls = []
    def set(self, key, value, ex=None):
        self.data[key] = value
        self.set_calls.append(key)
    def get(self, key):
        return self.data.get(key)
    def delete(self, key):
        self.data.pop(key, None)


@pytest.fixture()
def storage(tmp_path):
    redis_stub = FakeRedis()
    store = PodStorage(redis_client=redis_stub, lance_dir=str(tmp_path / "lance"))
    store.redis_stub = redis_stub
    return store


def test_put_and_get_roundtrip_l1(storage):
    storage.put("case:42", {"value": 42})
    assert storage.get("case:42") == {"value": 42}
    # L1 was written (redis stub received a set call)
    assert len(storage.redis_stub.set_calls) == 1


def test_l2_fallback_when_l1_empty(storage):
    storage.put("case:99", {"value": 99})
    # L1 "loses" the entry -> L2 Lance still serves it
    storage.redis_stub.data.clear()
    assert storage.get("case:99") == {"value": 99}


def test_traces_write_and_query(storage):
    storage.write_traces([
        {"trace_id": "t1", "case": "c1", "stage": "cache", "latency_ms": 0.1},
        {"trace_id": "t2", "case": "c2", "stage": "lookup", "latency_ms": 0.2},
    ])
    rows = storage.query_traces(stage="cache")
    assert any(r["trace_id"] == "t1" for r in rows)


def test_session_affinity_reuse_and_failover(storage):
    assert storage.kv_session("s1", "replica-0") == "replica-0"
    assert storage.kv_session("s1", "replica-0") == "replica-0"  # reuse
    assert storage.kv_session("s1", "replica-1") == "replica-1"  # failover
    stats = storage.stats()["session_affinity"]
    assert stats["reuses"] == 1 and stats["failovers"] == 1
