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


# --- regression tests for the L2 defects found in the 2026-09-20 review ----


def test_first_write_is_not_duplicated(storage):
    """create_table(data=rows) already stores rows; the old code add()ed them
    a second time, which made top-k return the same document twice."""
    storage.put("k1", {"v": 1})
    assert storage._open("kv").count_rows() == 1


def test_get_miss_does_not_create_a_table(storage):
    """A read must not write. The old get() ran the table-creating path and
    left a {'value': 'null'} placeholder behind for every miss."""
    assert storage.get("unbekannt") is None
    assert storage._tables() == []


def test_keys_and_filters_are_quoted(storage):
    storage.put("O'Brien", {"v": 2})
    assert storage.get("O'Brien") == {"v": 2}
    storage.write_traces([{"trace_id": "t", "case": "c", "stage": "O'Hara",
                           "latency_ms": 1.0}])
    assert len(storage.query_traces(stage="O'Hara")) == 1


def test_reput_replaces_instead_of_appending(storage):
    storage.put("k", {"v": 1})
    storage.put("k", {"v": 2})
    storage.redis_stub.data.clear()          # force the L2 read
    assert storage.get("k") == {"v": 2}
    assert storage._open("kv").count_rows() == 1


def test_documents_roundtrip_without_duplicates(storage):
    rows = [{"key": f"d{i}", "text": f"doc {i}", "vector": [float(i)] * 8}
            for i in range(5)]
    storage.add_documents("ns", rows)
    assert storage._open("docs_ns").count_rows() == 5
    hits = [r["key"] for r in storage.search_documents("ns", [1.0] * 8, top_k=2)]
    assert len(set(hits)) == 2               # two distinct documents, not d1 twice


def test_search_unknown_namespace_is_empty_not_a_dummy_schema(storage):
    """The old code created docs_<ns> with vector [0.0], pinning the column to
    one dimension and breaking every later add()."""
    assert storage.search_documents("ns", [0.1] * 8, top_k=3) == []
    assert storage._tables() == []


def test_more_than_ten_namespaces_are_listed(storage):
    """table_names() defaults to limit=10; a truncated list would make the
    write path try to create a table that already exists."""
    for i in range(12):
        storage.add_documents(f"ns{i}", [{"key": "x", "vector": [1.0] * 4}])
    assert len(storage._tables()) == 12


def test_l2_hit_warms_l1_back_up(storage):
    storage.put("k", {"v": 7})
    storage.redis_stub.data.clear()
    assert storage.get("k") == {"v": 7}
    assert len(storage.redis_stub.data) == 1


def test_empty_writes_are_noops(storage):
    storage.add_documents("ns", [])
    storage.write_traces([])
    assert storage._tables() == []
