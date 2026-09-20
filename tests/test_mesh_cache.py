"""MeshCache: what the principal scope does, and what invalidation does.

The module documented a mesh-announced invalidation while the code only
deleted the Redis key, and imported paho purely so the dependency would look
used. These tests pin down both halves: the shared delete, which always
happens, and the announcement, which happens only with a mesh endpoint.

They also pin down the principal scope in both directions. The old test
asserted `get(principal="tenant-b") is None` on a cache where nothing had
ever been written under tenant-b — true whether or not anything is isolated.
`research/benchmark_mesh_cache.py` recorded `principal_isolated: true` from
the same non-experiment. What is true is narrower and is tested as two
separate facts below: the class refuses a principal it was not built for,
and the entries themselves are not isolated at all.
"""
import json

import pytest

from neural_pods.mesh_cache import MeshCache, PrincipalRefused


class FakeRedis:
    def __init__(self, decode_responses=False):
        self.data = {}
        self.decode_responses = decode_responses

    def set(self, key, value, ex=None):
        self.data[key] = value

    def get(self, key):
        value = self.data.get(key)
        if value is None or self.decode_responses:
            return value
        return value.encode("utf-8") if isinstance(value, str) else value

    def delete(self, key):
        self.data.pop(key, None)


class FakeMesh:
    def __init__(self):
        self.published = []
        self.handlers = {}

    def publish_raw(self, topic, body, *, retain=False):
        self.published.append((topic, body))

    def subscribe(self, topic, handler):
        self.handlers[topic] = handler


def test_a_cache_refuses_a_principal_it_was_not_built_for():
    cache = MeshCache(redis_client=FakeRedis(), pod_id="pod-a",
                      principal="tenant-a")
    cache.put("q", {"v": 1})
    assert cache.get("q") == {"v": 1}
    with pytest.raises(PrincipalRefused, match="tenant-b"):
        cache.get("q", principal="tenant-b")
    assert cache.stats()["principal_refusals"] == 1


def test_a_second_principal_must_be_granted_explicitly():
    cache = MeshCache(redis_client=FakeRedis(), pod_id="pod-a",
                      principal="tenant-a",
                      allowed_principals=("tenant-b",))
    cache.put("q", {"v": 1}, principal="tenant-b")
    assert cache.get("q", principal="tenant-b") == {"v": 1}
    assert cache.get("q") is None          # tenant-a has no such entry
    assert cache.stats()["allowed_principals"] == ["tenant-a", "tenant-b"]
    assert cache.stats()["principal_refusals"] == 0


def test_the_principal_scope_is_not_a_boundary_and_says_so():
    """The entries of one principal are readable by anything holding the
    store, which is what `isolation: client-side key derivation` means.

    This is the test the benchmark should have run: build a second cache for
    the other principal against the SAME store and read the entry. It
    succeeds. An isolation claim that cannot fail is worth nothing, so the
    fact is pinned here rather than asserted away.
    """
    store = FakeRedis()
    private = MeshCache(redis_client=store, pod_id="pod-a",
                        principal="tenant-a-private", namespace="ns")
    private.put("secret", {"v": 9999})

    attacker = MeshCache(redis_client=store, pod_id="pod-b",
                         principal="tenant-a-private", namespace="ns")
    assert attacker.get("secret") == {"v": 9999}
    assert private.stats()["isolation"] == "client-side key derivation"


def test_an_undecodable_entry_is_not_counted_as_a_hit():
    store = FakeRedis()
    cache = MeshCache(redis_client=store, pod_id="pod-a")
    cache.put("q", {"v": 1})
    key = next(iter(store.data))
    store.data[key] = "{not json"

    assert cache.get("q") is None
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["decode_errors"] == 1


def test_get_works_with_decode_responses_enabled():
    """The hard-coded .decode() broke against a redis client configured to
    return str."""
    cache = MeshCache(redis_client=FakeRedis(decode_responses=True),
                      pod_id="pod-a")
    cache.put("q", {"v": 1})
    assert cache.get("q") == {"v": 1}


def test_without_a_mesh_an_invalidation_is_a_delete_and_says_so():
    cache = MeshCache(redis_client=FakeRedis(), pod_id="pod-a")
    cache.put("q", {"v": 1})
    cache.invalidate("q")
    assert cache.get("q") is None
    stats = cache.stats()
    assert stats["mesh_bound"] is False
    assert stats["invalidations_sent"] == 0


def test_with_a_mesh_the_invalidation_is_announced():
    mesh = FakeMesh()
    cache = MeshCache(redis_client=FakeRedis(), pod_id="pod-a",
                      principal="tenant-a", namespace="ns", mesh=mesh)
    cache.put("q", {"v": 1})
    cache.invalidate("q")

    assert cache.get("q") is None
    topic, body = mesh.published[0]
    assert topic == MeshCache.INVALIDATION_TOPIC
    assert body == {"namespace": "ns", "principal": "tenant-a",
                    "query_key": "q", "by": "pod-a"}
    assert cache.stats()["invalidations_sent"] == 1


def test_a_node_receives_other_nodes_invalidations_but_not_its_own():
    mesh = FakeMesh()
    cache = MeshCache(redis_client=FakeRedis(), pod_id="pod-a",
                      namespace="ns", mesh=mesh)
    dropped = []
    cache.subscribe_invalidations(dropped.append)
    handler = mesh.handlers[MeshCache.INVALIDATION_TOPIC]

    handler(MeshCache.INVALIDATION_TOPIC,
            {"body": {"namespace": "ns", "query_key": "q", "by": "pod-b"}})
    handler(MeshCache.INVALIDATION_TOPIC,
            {"body": {"namespace": "ns", "query_key": "q", "by": "pod-a"}})
    handler(MeshCache.INVALIDATION_TOPIC,
            {"body": {"namespace": "other", "query_key": "q", "by": "pod-b"}})

    assert [entry["by"] for entry in dropped] == ["pod-b"]
    assert cache.stats()["invalidations_received"] == 1


def test_subscribing_without_a_mesh_is_an_error_not_a_no_op():
    cache = MeshCache(redis_client=FakeRedis(), pod_id="pod-a")
    with pytest.raises(RuntimeError, match="no mesh endpoint"):
        cache.subscribe_invalidations()


def test_invalid_utf8_is_a_decode_error_not_an_exception():
    """The decode sat outside the guard, so bytes Redis could not decode
    raised UnicodeDecodeError at the caller. A cache that cannot read an
    entry must return a miss, not take the process with it."""
    store = FakeRedis()
    cache = MeshCache(redis_client=store, pod_id="pod-a")
    cache.put("q", {"v": 1})
    key = next(iter(store.data))
    store.data[key] = b"\xff\xfe not utf-8 at all"

    assert cache.get("q") is None
    stats = cache.stats()
    assert stats["decode_errors"] == 1
    assert stats["hits"] == 0
