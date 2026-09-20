from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from neural_pods.registry import Registry, InvalidState, verify_files, hash_files


def fixture(r, name="a", parents=None):
    o = r.origin("sap", name, "v1", {"value": 24})
    k = r.publish(name, {"value": 24}, parents or [o])
    p = r.artifact("lora", {"weights": "sha256:test"}, [k])
    return o, k, p


def test_closure_all_artifact_types_and_unrelated_survives(tmp_path):
    r = Registry(tmp_path / "r.db")
    o, k, p = fixture(r)
    _, _, unrelated = fixture(r, "b")
    v = r.artifact("vector", {}, [k])
    j = r.artifact("jspace", {}, [k])
    t = r.artifact("text", {}, [k])
    c = r.artifact("cache", {}, [p])
    a = r.commit(r.snapshot([c]), "24 days")["answer_id"]
    assert set(r.revoke(k)) == {k, p, v, j, t, c, a}
    assert not r.node(o)["revoked"]
    r.snapshot([unrelated])
    for item in [p, v, j, t, c, a]:
        with pytest.raises(InvalidState):
            r.snapshot([item])


def test_multiple_roots_deduplicate_and_revoke_only_closure(tmp_path):
    r = Registry(tmp_path / "r.db")
    roots = [r.origin("source", str(i), "v1", str(i)) for i in range(3)]
    k = r.publish("risk", {}, [roots[0], roots[0], roots[1], roots[2]])
    p = r.artifact("lora", {}, [k])
    _, _, other = fixture(r)
    assert r.roots(p) == sorted(roots)
    assert set(r.revoke(roots[1])) == {roots[1], k, p}
    r.snapshot([other])


def test_update_rejects_snapshot_cache_mixed_generations_and_restore(tmp_path):
    r = Registry(tmp_path / "r.db")
    o, k, p = fixture(r)
    snapshot = r.snapshot([p])
    cache = r.artifact("cache", {}, [p])
    new_origin = r.origin("sap", "a", "v2", {"value": 18})
    new = r.publish("a", {"value": 18}, [new_origin])
    p2 = r.artifact("lora", {"v": 2}, [new])
    for ids in [[p], [cache], [p, p2]]:
        with pytest.raises(InvalidState): r.snapshot(ids)
    with pytest.raises(InvalidState): r.commit(snapshot, "24 days")
    r.commit(r.snapshot([p2]), "18 days")
    restored = r.publish("a", {"value": 24}, [o])
    assert r.node(restored)["payload"]["generation"] == 3
    with pytest.raises(InvalidState): r.snapshot([p])
    p3 = r.artifact("lora", {"v": 3}, [restored])
    r.commit(r.snapshot([p3]), "24 days")


def test_no_resurrection_no_missing_parents_no_neural_state_without_knowledge(tmp_path):
    r = Registry(tmp_path / "r.db")
    o, k, p = fixture(r)
    with pytest.raises(InvalidState): r.artifact("lora", {}, [])
    with pytest.raises(InvalidState): r.artifact("lora", {}, [o])
    with pytest.raises(InvalidState): r.artifact("lora", {}, ["missing"])
    r.revoke(o)
    with pytest.raises(InvalidState): r.origin("sap", "a", "v1", {"value": 24})
    with pytest.raises(InvalidState): r.publish("a", {}, [o])


def test_acl_applies_through_all_ancestors(tmp_path):
    r = Registry(tmp_path / "r.db")
    o = r.origin("private", "1", "1", {}, acl=["alice"])
    k = r.publish("secret", {}, [o], principal="alice")
    p = r.artifact("lora", {}, [k], principal="alice")
    with pytest.raises(InvalidState): r.snapshot([p], principal="bob")
    r.commit(r.snapshot([p], principal="alice"), "allowed")


def test_origin_key_stable_and_unambiguous(tmp_path):
    r = Registry(tmp_path / "r.db")
    a = r.origin("a:b", "c", "1", {"b": 2, "a": 1})
    assert a == r.origin("a:b", "c", "1", {"a": 1, "b": 2})
    assert a != r.origin("a", "b:c", "1", {"a": 1, "b": 2})
    assert a != r.origin("a:b", "c", "2", {"a": 1, "b": 2})


def test_cross_connection_revoke_during_inference(tmp_path):
    path = tmp_path / "r.db"
    r = Registry(path)
    o, _, p = fixture(r)
    ready, revoked = threading.Event(), threading.Event()
    def inference():
        reader = Registry(path)
        s = reader.snapshot([p])
        ready.set()
        assert revoked.wait(5)
        with pytest.raises(InvalidState): reader.commit(s, "stale neural output")
        reader.close()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(inference)
        assert ready.wait(5)
        r.revoke(o)
        revoked.set()
        future.result()
    assert r.db.execute("SELECT count(*) FROM events WHERE action='commit'").fetchone()[0] == 0


def test_artifact_tampering(tmp_path):
    p = tmp_path / "adapter.safetensors"
    p.write_bytes(b"original")
    expected = hash_files(tmp_path)
    verify_files(tmp_path, expected)
    p.write_bytes(b"changed")
    with pytest.raises(InvalidState): verify_files(tmp_path, expected)


def test_events_can_be_recorded_from_another_thread():
    """The registry is written from the threads the rest of the system runs
    on; without check_same_thread=False the first such write raises
    sqlite3.ProgrammingError."""
    import threading

    registry = Registry(":memory:")
    errors: list[BaseException] = []

    def record(index: int):
        try:
            with registry.transaction():
                registry.record_event("probe", {"index": index})
        except BaseException as error:        # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=record, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len([e for e in registry.events() if e["action"] == "probe"]) == 8


def test_record_event_is_the_public_name_of_event():
    registry = Registry(":memory:")
    with registry.transaction():
        registry.record_event("public", {"a": 1})
    assert registry.events(action="public")[0]["payload"] == {"a": 1}
    assert registry._event.__func__ is registry.record_event.__func__


def test_nested_transactions_commit_once():
    registry = Registry(":memory:")
    with registry.transaction():
        registry.record_event("outer", {})
        with registry.transaction():        # re-entrant, same thread
            registry.record_event("inner", {})
    assert {e["action"] for e in registry.events()} >= {"outer", "inner"}
