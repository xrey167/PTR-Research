from neural_pods.snapshot_store import SnapshotStore


def test_snapshot_roundtrip_integrity_and_compaction(tmp_path):
    store = SnapshotStore(tmp_path)
    store.append_wal("research", {"revision": 1, "op": "upsert"})
    store.write("research", 1, [{"key": "a", "value": 1}])
    store.write("research", 2, [{"key": "a", "value": 2}])
    ref, rows = store.read("research")
    assert ref.revision == 2 and rows[0]["value"] == 2
    assert store.read("research", 1)[1][0]["value"] == 1
    assert store.compact("research", keep=1) == 1
