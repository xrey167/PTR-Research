from neural_pods.local_search import LocalSearchBackend
from neural_pods.pod_cache import PodCache
from concurrent.futures import ThreadPoolExecutor
import time


def test_cache_isolates_namespace_type_and_tags_and_reports_hits(tmp_path):
    backend = LocalSearchBackend(tmp_path / "s.sqlite")
    backend.create_namespace("pods")
    backend.upsert("pods", "ctx", text="Muller delivery", metadata={"type": "context", "tags": ["supplier"], "status": "active"})
    backend.upsert("pods", "math", text="Muller delivery", metadata={"type": "math", "tags": ["calculation"], "status": "active"})
    cache = PodCache(max_entries=8)
    first = cache.search(backend, "pods", text="Muller", pod_type="context", tags=["supplier"])
    second = cache.search(backend, "pods", text="Muller", pod_type="context", tags=["supplier"])
    assert [x.key for x in first] == ["ctx"] and [x.key for x in second] == ["ctx"]
    assert cache.stats()["hits"] == 1
    assert [x.key for x in cache.search(backend, "pods", text="Muller", pod_type="math", tags=["calculation"])] == ["math"]
    assert cache.stats()["entries"] == 2
    assert cache.invalidate(namespace="pods", pod_type="context") == 1
    backend.close()


def test_cache_key_isolates_vector_filters_and_principal(tmp_path):
    backend = LocalSearchBackend(tmp_path / "isolated.sqlite")
    backend.create_namespace("pods")
    backend.upsert("pods", "a", text="same", vector=[1, 0], metadata={"status": "active", "acl": ["a"]})
    backend.upsert("pods", "b", text="same", vector=[0, 1], metadata={"status": "active", "acl": ["b"]})
    cache = PodCache()
    assert [x.key for x in cache.search(backend, "pods", text="same", vector=[1, 0], principal="a")] == ["a"]
    assert [x.key for x in cache.search(backend, "pods", text="same", vector=[0, 1], principal="b")] == ["b"]
    assert cache.stats()["entries"] == 2
    backend.close()


def test_cache_single_flight_coalesces_bursty_reads():
    class Backend:
        def __init__(self): self.calls = 0
        def search(self, *args, **kwargs):
            self.calls += 1; time.sleep(0.02); return ["ok"]
    backend = Backend(); cache = PodCache()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cache.search(backend, "n", text="q"), range(8)))
    assert backend.calls == 1 and all(result == ["ok"] for result in results)


def test_namespace_revision_prevents_stale_cached_results(tmp_path):
    backend = LocalSearchBackend(tmp_path / "revision.sqlite")
    backend.create_namespace("pods")
    backend.upsert("pods", "a", text="old", metadata={"status": "active"})
    cache = PodCache()
    assert [x.key for x in cache.search(backend, "pods", text="old")] == ["a"]
    backend.upsert("pods", "b", text="new", metadata={"status": "active"})
    assert [x.key for x in cache.search(backend, "pods", text="new")] == ["b"]
    assert cache.stats()["misses"] == 2


def test_explicit_hot_pin_survives_lru_eviction_and_unpin_restores_bound(tmp_path):
    class Backend:
        def __init__(self): self.calls = 0
        def namespace_metadata(self, namespace): return {"revision": 1}
        def search(self, namespace, **kwargs):
            self.calls += 1
            return [{"key": kwargs["text"]}]

    backend = Backend(); cache = PodCache(max_entries=2, ttl_seconds=60)
    cache.pin_query(backend, "pods", text="hot")
    cache.search(backend, "pods", text="cold-1")
    cache.search(backend, "pods", text="cold-2")
    assert cache.stats()["pinned"] == 1
    assert cache.search(backend, "pods", text="hot")[0]["key"] == "hot"
    assert cache.stats()["hits"] >= 1
    assert cache.unpin(namespace="pods") == 1
    cache.search(backend, "pods", text="cold-3")
    assert cache.stats()["entries"] <= 2
    assert cache.stats()["evictions"] >= 1


def test_revision_change_drops_pinned_old_generation():
    class Backend:
        def __init__(self): self.revision = 1
        def namespace_metadata(self, namespace): return {"revision": self.revision}
        def search(self, namespace, **kwargs): return [{"key": kwargs["text"], "revision": self.revision}]

    backend = Backend(); cache = PodCache(max_entries=4, ttl_seconds=60)
    cache.pin_query(backend, "pods", text="hot")
    assert cache.stats()["pinned"] == 1
    backend.revision = 2
    assert cache.search(backend, "pods", text="new")[0]["revision"] == 2
    assert cache.stats()["pinned"] == 0
    assert cache.stats()["entries"] == 1
