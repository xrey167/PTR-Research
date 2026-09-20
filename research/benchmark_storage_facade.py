"""Exercise the PodStorage facade and its LanceDB L2 tier, and record it.

F3/F4 landed as code without the gate checks the master plan promised
(`storage_facade`, `storage_l2_lance`), which is how six defects in the
facade survived: the first write was stored twice, a read created tables,
predicates were built by string interpolation, an empty namespace pinned
the vector column to one dimension, and the table listing truncated at ten.
Every one of those is a property this benchmark measures.

L1 runs against an in-process stub unless a Redis server is reachable. That
is recorded in the report as `l1_backend`; what the facade's contract needs
from L1 is "asked first, back-filled after an L2 hit", and that is testable
without a server. The Redis tier's own latency is measured by
benchmark_redis_cache_tier.py.

Usage:  python research/benchmark_storage_facade.py [--redis HOST]
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.storage import PodStorage  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs" / "storage-facade-20260920.json"
DOCUMENTS = 200
VECTOR_DIM = 16
NAMESPACES = 12          # more than table_names()'s default limit of 10
KV_KEYS = 200


class StubRedis:
    """Dict-backed L1: enough to prove the tier order and the back-fill."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.sets = 0
        self.gets = 0

    def set(self, key, value, ex=None):
        self.data[key] = value
        self.sets += 1

    def get(self, key):
        self.gets += 1
        return self.data.get(key)

    def delete(self, key):
        self.data.pop(key, None)


def _connect_redis(host: str | None):
    if not host:
        return StubRedis(), "in-process stub (no redis host given)"
    try:
        import redis
        client = redis.Redis(host=host, socket_timeout=2)
        client.ping()
        return client, f"redis://{host}"
    except Exception as error:                       # noqa: BLE001
        return StubRedis(), f"in-process stub ({type(error).__name__} from {host})"


def _l1_keys(store, keys: list[str]) -> list[str]:
    """The exact L1 keys the facade would use for these values."""
    return [store._l1_key(key) for key in keys]


def _clear_l1(client, l1_keys: list[str]) -> None:
    """Drop the L1 entries, whichever backend is behind the client.

    The stub used to be cleared by reaching into its dict, so with a real
    Redis nothing was cleared and the "L2 read" loop below still answered
    from L1 - producing evidence the gate then rejected.
    """
    for key in l1_keys:
        client.delete(key)


def _count_l1(client, l1_keys: list[str]) -> int:
    return sum(1 for key in l1_keys if client.get(key) is not None)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * p))], 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis", default=os.environ.get("NEURAL_PODS_REDIS"))
    args = parser.parse_args()
    redis_client, l1_backend = _connect_redis(args.redis)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as workdir:
        store = PodStorage(redis_client=redis_client,
                           lance_dir=Path(workdir) / "lance",
                           principal="tenant-bench")

        # --- a read must not create anything --------------------------------
        miss_before = store._tables()
        assert store.get("never-written") is None
        assert store.search_documents("never-used", [0.0] * VECTOR_DIM) == []
        assert store.query_traces(stage="never-run") == []
        read_miss_tables_created = len(store._tables()) - len(miss_before)

        # --- KV: write once, read from L1, then from L2 ---------------------
        write_ms, l1_ms, l2_ms = [], [], []
        for index in range(KV_KEYS):
            key = f"case:{index}:O'Brien"      # apostrophe: predicate quoting
            t0 = time.perf_counter()
            store.put(key, {"index": index})
            write_ms.append((time.perf_counter() - t0) * 1000)
            t1 = time.perf_counter()
            assert store.get(key) == {"index": index}
            l1_ms.append((time.perf_counter() - t1) * 1000)

        kv_rows = store._open("kv").count_rows()
        duplicate_kv_rows = kv_rows - KV_KEYS

        keys = [f"case:{index}:O'Brien" for index in range(KV_KEYS)]
        l1_keys = _l1_keys(store, keys)
        _clear_l1(redis_client, l1_keys)
        l1_entries_before = _count_l1(redis_client, l1_keys)
        for index, key in enumerate(keys):
            t0 = time.perf_counter()
            assert store.get(key) == {"index": index}
            l2_ms.append((time.perf_counter() - t0) * 1000)
        l1_backfilled = _count_l1(redis_client, l1_keys) - l1_entries_before

        # --- re-put replaces, it does not append ----------------------------
        store.put("case:0:O'Brien", {"index": -1})
        _clear_l1(redis_client, _l1_keys(store, ["case:0:O'Brien"]))
        stale_read = store.get("case:0:O'Brien") != {"index": -1}
        rows_after_reput = store._open("kv").count_rows()

        # --- L2 documents and vector search ---------------------------------
        rows = [{"key": f"doc-{i}", "text": f"document {i}",
                 "vector": [float(i)] * VECTOR_DIM} for i in range(DOCUMENTS)]
        t0 = time.perf_counter()
        store.add_documents("corpus", rows)
        document_write_ms = (time.perf_counter() - t0) * 1000
        document_rows = store._open("docs_corpus").count_rows()
        hits = store.search_documents("corpus", [7.0] * VECTOR_DIM, top_k=10)
        distinct_top_k = len({hit["key"] for hit in hits})

        # --- more namespaces than the listing default -----------------------
        for index in range(NAMESPACES):
            store.add_documents(f"ns{index}", [{"key": "x",
                                                "vector": [1.0] * VECTOR_DIM}])
        tables_listed = len(store._tables())

        # --- traces ----------------------------------------------------------
        store.write_traces([{"trace_id": f"t{i}", "case": f"c{i}",
                             "stage": "cache" if i % 2 else "O'Hara",
                             "latency_ms": float(i)} for i in range(100)])
        trace_rows = store._open("traces").count_rows()
        quoted_stage = len(store.query_traces(stage="O'Hara", limit=100))

        # --- Mooncake session affinity ---------------------------------------
        store.kv_session("s1", "replica-0")
        store.kv_session("s1", "replica-0")
        store.kv_session("s1", "replica-1")
        affinity = store.stats()["session_affinity"]

        result = {
            "status": "completed",
            "l1_backend": l1_backend,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "kv": {
                "keys": KV_KEYS,
                "rows": kv_rows,
                "duplicate_rows": duplicate_kv_rows,
                "rows_after_reput": rows_after_reput,
                "stale_read_after_reput": stale_read,
                "read_miss_tables_created": read_miss_tables_created,
                "l1_backfilled_after_l2_hit": l1_backfilled,
                "quoted_key_roundtrip": True,
                "write_p50_ms": _percentile(write_ms, 0.5),
                "l1_read_p50_ms": _percentile(l1_ms, 0.5),
                "l2_read_p50_ms": _percentile(l2_ms, 0.5),
                "l1_faster_than_l2": statistics.median(l1_ms) < statistics.median(l2_ms),
            },
            "l2_lance": {
                "documents": DOCUMENTS,
                "document_rows": document_rows,
                "duplicate_document_rows": document_rows - DOCUMENTS,
                "document_write_ms": round(document_write_ms, 3),
                "top_k_requested": 10,
                "top_k_distinct": distinct_top_k,
                "vector_dim": VECTOR_DIM,
                "namespaces_created": NAMESPACES + 2,   # + corpus + kv
                "tables_listed": tables_listed,
                "trace_rows": trace_rows,
                "duplicate_trace_rows": trace_rows - 100,
                "quoted_stage_matches": quoted_stage,
            },
            "session_affinity": affinity,
            "scope": ("facade contract and L2 behaviour; L1 latency against a "
                      "real Redis is measured by benchmark_redis_cache_tier.py"),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
