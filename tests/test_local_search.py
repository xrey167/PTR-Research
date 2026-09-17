from concurrent.futures import ThreadPoolExecutor
import math
import pytest

from neural_pods.local_search import LocalSearchBackend


def test_local_backend_fusion_filters_and_provenance(tmp_path):
    backend = LocalSearchBackend(tmp_path / "search.sqlite")
    backend.create_namespace("pods")
    backend.upsert("pods", "a", text="Müller X12 lead time 24 days", vector=[1, 0],
                   metadata={"knowledge_key": "supplier:muller:x12:lead_time", "generation_key": "g8",
                             "origin_keys": ["o17"], "status": "active", "acl": ["buyer"], "tags": ["supplier"]})
    backend.upsert("pods", "b", text="Müller X12 lead time 18 days", vector=[.8, .2],
                   metadata={"knowledge_key": "supplier:muller:x12:lead_time", "generation_key": "g7",
                             "origin_keys": ["o17"], "status": "active", "acl": ["buyer"], "tags": ["supplier"]})
    backend.upsert("pods", "secret", text="Müller X12 lead time 1 day", vector=[1, 0],
                   metadata={"status": "active", "acl": ["admin"]})
    hits = backend.search("pods", text="How long is Müller X12?", vector=[1, 0],
                          filters={"generation_key": "g8", "origin_keys": {"contains": ["o17"]}},
                          principal="buyer")
    assert [hit.key for hit in hits] == ["a"]
    assert backend.search("pods", text="", filters={}, principal="buyer", top_k=10)[0].key in {"a", "b"}
    assert all(hit.key != "secret" for hit in backend.search("pods", text="Müller", principal="buyer"))
    assert backend.search("pods", regex=r"24 days", principal="buyer")[0].key == "a"
    backend.upsert("pods", "surprise", text="new contradiction", vector=[1, 0],
                   metadata={"status": "active", "surprise_gate": {"novelty_threshold": .7, "contradiction_threshold": .3},
                             "novelty_score": .8, "contradiction_score": .8})
    assert all(hit.key != "surprise" for hit in backend.search("pods", text="new", principal="buyer"))
    backend.close()


def test_namespace_prewarm_reports_hot_cache(tmp_path):
    backend = LocalSearchBackend(tmp_path / "warm.sqlite")
    backend.create_namespace("pods")
    backend.upsert("pods", "a", text="warm", metadata={"status": "active"})
    result = backend.prewarm("pods")
    assert result["rows_warmed"] == 1 and result["cache_temperature"] == "hot"
    assert backend.namespace_metadata("pods")["index"]["status"] == "up-to-date"
    backend.close()


def test_build_persistent_vector_index(tmp_path):
    backend = LocalSearchBackend(tmp_path / "vectors.sqlite")
    backend.create_namespace("pods")
    backend.upsert("pods", "a", text="a", vector=[1, 0], metadata={"status": "active"})
    backend.upsert("pods", "b", text="b", vector=[0, 1], metadata={"status": "active"})
    stats = backend.build_vector_index("pods", path=tmp_path / "pods-index")
    assert stats["vectors"] == 2 and stats["storage"] in {"hnswlib", "numpy-npz"}
    backend.close()


def test_copy_on_write_branch_and_parallel_stateless_readers(tmp_path):
    backend = LocalSearchBackend(tmp_path / "search.sqlite")
    backend.create_namespace("n")
    backend.upsert("n", "base", text="base knowledge", vector=[1, 0], metadata={"status": "active"})
    backend.branch("n", target="experiment")
    backend.upsert("n", "new", text="experimental knowledge", vector=[0, 1], branch="experiment",
                   metadata={"status": "active"})
    assert {x.key for x in backend.search("n", branch="main", text="knowledge", top_k=10)} == {"base"}
    assert {x.key for x in backend.search("n", branch="experiment", text="knowledge", top_k=10)} == {"base", "new"}
    backend.pin("n", "experiment")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: backend.search("n", branch="experiment", vector=[0, 1]), range(32)))
    assert all(result and result[0].key == "new" for result in results)
    backend.unpin("n", "experiment")
    assert backend.stats("n", "experiment")["pinned"] is False
    backend.close()


def test_parent_write_invalidates_descendant_pins_and_rejects_bad_vectors(tmp_path):
    backend = LocalSearchBackend(tmp_path / "search.sqlite")
    backend.create_namespace("n")
    backend.upsert("n", "base", text="old", vector=[1, 0], metadata={"status": "active"})
    backend.branch("n", target="child")
    assert backend.pin("n", "child") == 1
    assert backend.stats("n", "child")["pinned"]
    backend.upsert("n", "new", text="parent update", vector=[0, 1], metadata={"status": "active"})
    assert backend.stats("n", "child")["pinned"] is False
    assert "new" in {hit.key for hit in backend.search("n", branch="child", text="parent")}
    with pytest.raises(ValueError):
        backend.upsert("n", "bad", vector=[math.nan], metadata={"status": "active"})
    backend.close()


def test_turbopuffer_style_multi_query_rrf_and_filters(tmp_path):
    backend = LocalSearchBackend(tmp_path / "query.sqlite")
    backend.create_namespace("pods")
    backend.upsert("pods", "a", text="Muller X12 lead time 24 days", vector=[1, 0],
                   metadata={"status": "active", "kind": "fact", "tags": ["supplier"]})
    backend.upsert("pods", "b", text="Muller X12 supplier risk high", vector=[0, 1],
                   metadata={"status": "active", "kind": "risk", "tags": ["supplier"]})
    backend.upsert("pods", "secret", text="Muller X12 1 day", vector=[1, 0],
                   metadata={"status": "active", "acl": ["admin"], "kind": "fact"})
    out = backend.query("pods", {"queries": [
        {"rank_by": ["text", "BM25", "Muller X12"], "filters": ["And", [["kind", "Eq", "fact"]]], "limit": {"total": 5}},
        {"rank_by": ["vector", "ANN", [1, 0]], "filters": {"tags": {"contains": ["supplier"]}}, "top_k": 5},
    ], "rerank_by": ["RRF"], "top_k": 2}, principal="buyer")
    assert [row["id"] for row in out["rows"]] == ["a", "b"]
    nested = backend.query("pods", {"rank_by": ["text", "BM25", "Muller"],
        "filters": ["And", [["status", "Eq", "active"], ["Or", [["kind", "Eq", "risk"], ["kind", "Eq", "fact"]]]]], "limit": 10}, principal="buyer")
    assert {row["id"] for row in nested["rows"]} == {"a", "b"}
    backend.close()


def test_namespace_metadata_and_read_only_barrier(tmp_path):
    backend = LocalSearchBackend(tmp_path / "metadata.sqlite")
    backend.create_namespace("pods")
    before = backend.namespace_metadata("pods")
    assert before["approx_row_count"] == 0 and before["index"]["status"] == "up-to-date"
    backend.upsert("pods", "a", text="fact", metadata={"status": "active"})
    meta = backend.namespace_metadata("pods")
    assert meta["approx_row_count"] == 1 and meta["last_write_at"]
    backend.update_namespace_metadata("pods", pinning={"replicas": 2}, read_only=True)
    assert backend.namespace_metadata("pods")["pinning"] == {"replicas": 2}
    with pytest.raises(PermissionError):
        backend.upsert("pods", "b", text="blocked", metadata={"status": "active"})
    backend.update_namespace_metadata("pods", read_only=False, pinning=None)
    backend.upsert("pods", "b", text="allowed", metadata={"status": "active"})
    backend.close()


def test_local_native_embed_query_without_provider_api(tmp_path):
    def embed(text):
        return [1.0, 0.0] if "walrus" in text.casefold() else [0.0, 1.0]
    backend = LocalSearchBackend(tmp_path / "embed.sqlite", embedder=embed)
    backend.create_namespace("docs")
    backend.configure_schema("docs", {"text": {"type": "string", "embed": {"model": "local", "dims": 2}}})
    backend.upsert("docs", "walrus", text="arctic walrus", vector=[1, 0], metadata={"status": "active"})
    backend.upsert("docs", "fish", text="deep fish", metadata={"status": "active"})
    result = backend.query("docs", {"rank_by": ["text", "ANN", ["Embed", "arctic sea walrus"]], "limit": 1})
    assert result["rows"][0]["id"] == "walrus"
    assert backend.namespace_metadata("docs")["schema"]["text"]["embed"]["model"] == "local"
    with pytest.raises(ValueError, match="model mismatch"):
        backend.query("docs", {"rank_by": ["text", "ANN", ["Embed", "walrus", {"model": "other"}]], "limit": 1})
    backend.close()


def test_query_attribute_projection_and_aggregations(tmp_path):
    backend = LocalSearchBackend(tmp_path / "agg.sqlite")
    backend.create_namespace("docs")
    for key, group, value in (("a", "x", 2), ("b", "x", 3), ("c", "y", 5)):
        backend.upsert("docs", key, text=group, metadata={"status": "active", "group": group, "value": value})
    rows = backend.query("docs", {"rank_by": ["text", "BM25", "x"], "limit": 10,
                                   "include_attributes": ["group"], "exclude_attributes": ["group"]})["rows"]
    assert rows and set(rows[0]) == {"id", "$dist"}
    assert backend.query("docs", {"aggregate_by": {"count": ["Count"], "sum": ["Sum", "value"]}})["aggregations"] == {"count": 3, "sum": 10.0}
    groups = backend.query("docs", {"aggregate_by": {"count": ["Count"]}, "group_by": ["group"], "limit": 10})["aggregation_groups"]
    assert {x["group"]: x["count"] for x in groups} == {"x": 2, "y": 1}
    backend.close()


def test_explicit_namespace_reembedding_is_versioned(tmp_path):
    calls = []
    def embed(text):
        calls.append(text)
        return [1.0, 0.0] if "old" in text else [0.0, 1.0]
    backend = LocalSearchBackend(tmp_path / "reembed.sqlite", embedder=embed)
    backend.create_namespace("docs")
    backend.configure_schema("docs", {"text": {"embed": "model-v1"}})
    backend.upsert("docs", "a", text="old fact", vector=[0, 1], metadata={"status": "active", "revision": 4})
    backend.pin("docs")
    report = backend.reembed_namespace("docs")
    assert report["rows_reembedded"] == 1 and calls == ["old fact"]
    assert backend.stats("docs")["pinned"] is False
    assert backend.search("docs", vector=[1, 0])[0].key == "a"
    assert backend._rows("docs", "main")["a"]["metadata"]["revision"] == 5
    backend.close()


def test_weighted_sum_product_hybrid_rank(tmp_path):
    backend = LocalSearchBackend(tmp_path / "sum.sqlite")
    backend.create_namespace("docs")
    backend.upsert("docs", "semantic", text="unrelated wording", vector=[1, 0], metadata={"status": "active"})
    backend.upsert("docs", "lexical", text="walrus exact", vector=[0, 1], metadata={"status": "active"})
    result = backend.query("docs", {"rank_by": ["Sum", [["Product", 2, ["text", "BM25", "walrus"]], ["text", "ANN", [1, 0]]]], "limit": 2})
    assert result["rows"][0]["id"] == "lexical"
    backend.close()
