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

STRUCTURE, and two things the split fixed.

`collect()` exercises the facade and returns raw OBSERVATIONS; `summarise()`
turns them into the verdict the gate reads, and is pure. That boundary is
not cosmetic here:

  * The checks used to be `assert` statements inside the measurement. An
    assert that fires kills the run, so the one case worth recording — the
    facade got it wrong — produced a traceback and NO evidence. The gate saw
    a missing file, which it reports as "no evidence" rather than "evidence
    says no". Those are different findings and they need different fixes.
    Every assert is now an observation the summary judges.
  * `quoted_key_roundtrip` was written into the report as the literal `True`.
    The gate's `quoted_key_roundtrip is True` therefore asserted nothing at
    all — the same defect as `reflex_frames_valid == 132`. It is now a count
    of the round-trips that actually returned what was written, and the
    boolean follows from the count.

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
from research.evidence import write as write_evidence  # noqa: E402

#: The modules these numbers are evidence ABOUT. research/evidence.py
#: hashes them into the report, and the gate refuses the file once any
#: of them changes: a measurement of code that no longer exists is not
#: evidence, however carefully it was recorded.
SUBJECT = [
    "neural_pods/storage.py",
]

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


def summarise(obs: dict) -> dict:
    """Turn the observations into the evidence the gate reads. Pure.

    Every field the gate compares is derived here, from numbers `collect()`
    measured. Nothing in this function can reach Redis, LanceDB or a clock,
    which is what makes it possible to ask it what it says about a facade
    that got things WRONG — the case the asserts used to make unreportable.
    """
    kv_keys = obs["kv_keys"]
    l1_ms, l2_ms = obs["l1_read_ms"], obs["l2_read_ms"]
    return {
        "status": "completed",
        "l1_backend": obs["l1_backend"],
        "elapsed_s": obs["elapsed_s"],
        "kv": {
            "keys": kv_keys,
            "rows": obs["kv_rows"],
            # A first write stored twice was one of the six defects.
            "duplicate_rows": obs["kv_rows"] - kv_keys,
            "rows_after_reput": obs["rows_after_reput"],
            # A re-put must replace. Reading the old value back means the
            # facade appended and the reader found the stale row first.
            "stale_read_after_reput": obs["reput_value_read"] != obs["reput_value_written"],
            # A read must not create tables.
            "read_miss_tables_created": (obs["tables_after_read_miss"]
                                         - obs["tables_before"]),
            "read_miss_returned_nothing": (obs["read_miss_get"] is None
                                           and obs["read_miss_search"] == 0
                                           and obs["read_miss_traces"] == 0),
            "l1_backfilled_after_l2_hit": (obs["l1_entries_after_backfill"]
                                           - obs["l1_entries_before_backfill"]),
            # Counted, not asserted: how many keys containing an apostrophe
            # came back as written. The literal `True` that used to stand
            # here made the gate's check on it unfalsifiable.
            "quoted_key_roundtrips_ok": obs["quoted_key_roundtrips_ok"],
            "quoted_key_roundtrips_attempted": obs["quoted_key_roundtrips_attempted"],
            "quoted_key_roundtrip": (obs["quoted_key_roundtrips_ok"]
                                     == obs["quoted_key_roundtrips_attempted"]
                                     and obs["quoted_key_roundtrips_attempted"] > 0),
            "write_p50_ms": _percentile(obs["write_ms"], 0.5),
            "l1_read_p50_ms": _percentile(l1_ms, 0.5),
            "l2_read_p50_ms": _percentile(l2_ms, 0.5),
            "l1_faster_than_l2": (statistics.median(l1_ms)
                                  < statistics.median(l2_ms)),
            # A store that already held data before the l1_eligible column
            # existed. Without the schema migration every write against it
            # raises; without the NULL branch in get() its rows never reach
            # L1 again. Both halves, measured on one table.
            "legacy_table_migrated": (obs["legacy_table_had_flag"] is False
                                      and obs["legacy_write_ok"] is True
                                      and obs["legacy_table_has_flag_after"] is True),
            "legacy_write_error": obs["legacy_write_error"],
            "legacy_row_readable": obs["legacy_row_value"] == {"index": 0},
            "legacy_row_backfilled_l1": obs["legacy_row_backfilled_l1"],
        },
        "l2_lance": {
            "documents": obs["documents"],
            "document_rows": obs["document_rows"],
            "duplicate_document_rows": obs["document_rows"] - obs["documents"],
            "document_write_ms": round(obs["document_write_ms"], 3),
            "top_k_requested": obs["top_k_requested"],
            "top_k_distinct": obs["top_k_distinct"],
            "vector_dim": VECTOR_DIM,
            "namespaces_created": obs["namespaces_created"],
            # The listing truncated at ten, so more namespaces than the
            # default limit is the point of this number.
            "tables_listed": obs["tables_listed"],
            "trace_rows": obs["trace_rows"],
            "duplicate_trace_rows": obs["trace_rows"] - obs["traces_written"],
            "quoted_stage_matches": obs["quoted_stage_matches"],
        },
        "session_affinity": obs["session_affinity"],
        "scope": ("facade contract and L2 behaviour; L1 latency against a "
                  "real Redis is measured by benchmark_redis_cache_tier.py"),
    }


def collect(*, redis_host: str | None = None) -> dict:
    """Exercise the facade and return raw observations. No verdicts here."""
    redis_client, l1_backend = _connect_redis(redis_host)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as workdir:
        store = PodStorage(redis_client=redis_client,
                           lance_dir=Path(workdir) / "lance",
                           principal="tenant-bench")

        # --- a read must not create anything --------------------------------
        tables_before = len(store._tables())
        read_miss_get = store.get("never-written")
        read_miss_search = len(store.search_documents("never-used",
                                                      [0.0] * VECTOR_DIM))
        read_miss_traces = len(store.query_traces(stage="never-run"))
        tables_after_read_miss = len(store._tables())

        # --- KV: write once, read from L1, then from L2 ---------------------
        write_ms, l1_ms, l2_ms = [], [], []
        # Counted, not derived from KV_KEYS: every key is read twice (once
        # from L1, once from L2 after the L1 entries are cleared), so an
        # expectation computed from the key count alone is off by a factor
        # of two — as the first run of this split showed.
        roundtrips_ok = roundtrips_attempted = 0
        for index in range(KV_KEYS):
            key = f"case:{index}:O'Brien"      # apostrophe: predicate quoting
            t0 = time.perf_counter()
            store.put(key, {"index": index})
            write_ms.append((time.perf_counter() - t0) * 1000)
            t1 = time.perf_counter()
            value = store.get(key)
            l1_ms.append((time.perf_counter() - t1) * 1000)
            roundtrips_attempted += 1
            roundtrips_ok += 1 if value == {"index": index} else 0

        kv_rows = store._open("kv").count_rows()

        keys = [f"case:{index}:O'Brien" for index in range(KV_KEYS)]
        l1_keys = _l1_keys(store, keys)
        _clear_l1(redis_client, l1_keys)
        l1_entries_before = _count_l1(redis_client, l1_keys)
        for index, key in enumerate(keys):
            t0 = time.perf_counter()
            value = store.get(key)
            l2_ms.append((time.perf_counter() - t0) * 1000)
            roundtrips_attempted += 1
            roundtrips_ok += 1 if value == {"index": index} else 0
        l1_entries_after = _count_l1(redis_client, l1_keys)

        # --- re-put replaces, it does not append ----------------------------
        reput_written = {"index": -1}
        store.put("case:0:O'Brien", reput_written)
        _clear_l1(redis_client, _l1_keys(store, ["case:0:O'Brien"]))
        reput_read = store.get("case:0:O'Brien")
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
        traces_written = 100
        store.write_traces([{"trace_id": f"t{i}", "case": f"c{i}",
                             "stage": "cache" if i % 2 else "O'Hara",
                             "latency_ms": float(i)}
                            for i in range(traces_written)])
        trace_rows = store._open("traces").count_rows()
        quoted_stage = len(store.query_traces(stage="O'Hara", limit=100))

        # --- a kv table that predates the l1_eligible column ----------------
        # The reason the gate never saw this case: every run of this benchmark
        # starts from a fresh TemporaryDirectory, so it only ever meets a
        # table it created itself. A grown store is the one on the server.
        # lancedb refuses an unknown column outright ("Field 'l1_eligible'
        # not found in target schema"), so without the migration every put()
        # against such a table raises.
        legacy_dir = Path(workdir) / "legacy"
        legacy_redis = StubRedis()
        legacy = PodStorage(redis_client=legacy_redis, lance_dir=legacy_dir,
                            principal="tenant-bench")
        legacy._lance_db().create_table("kv", data=[{
            "key": "old-row", "value": json.dumps({"index": 0}),
            "principal": "tenant-bench", "ts": 0.0}])
        legacy_had_flag = "l1_eligible" in legacy._open("kv").schema.names
        try:
            legacy.put("after-migration", {"index": 1})
            legacy_write_ok = True
        except Exception as error:               # noqa: BLE001 - recorded
            legacy_write_ok = False
            legacy_error = repr(error)
        else:
            legacy_error = None
        legacy_has_flag = "l1_eligible" in legacy._open("kv").schema.names
        legacy_redis.data.clear()
        legacy_value = legacy.get("old-row")
        legacy_backfilled = len(legacy_redis.data)

        # --- Mooncake session affinity ---------------------------------------
        store.kv_session("s1", "replica-0")
        store.kv_session("s1", "replica-0")
        store.kv_session("s1", "replica-1")
        affinity = store.stats()["session_affinity"]

        return {
            "l1_backend": l1_backend,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "kv_keys": KV_KEYS,
            "kv_rows": kv_rows,
            "tables_before": tables_before,
            "tables_after_read_miss": tables_after_read_miss,
            "read_miss_get": read_miss_get,
            "read_miss_search": read_miss_search,
            "read_miss_traces": read_miss_traces,
            "quoted_key_roundtrips_ok": roundtrips_ok,
            "quoted_key_roundtrips_attempted": roundtrips_attempted,
            "write_ms": write_ms,
            "l1_read_ms": l1_ms,
            "l2_read_ms": l2_ms,
            "l1_entries_before_backfill": l1_entries_before,
            "l1_entries_after_backfill": l1_entries_after,
            "reput_value_written": reput_written,
            "reput_value_read": reput_read,
            "rows_after_reput": rows_after_reput,
            "documents": DOCUMENTS,
            "document_rows": document_rows,
            "document_write_ms": document_write_ms,
            "top_k_requested": 10,
            "top_k_distinct": distinct_top_k,
            "namespaces_created": NAMESPACES + 2,   # + corpus + kv
            "tables_listed": tables_listed,
            "traces_written": traces_written,
            "trace_rows": trace_rows,
            "quoted_stage_matches": quoted_stage,
            "session_affinity": affinity,
            "legacy_table_had_flag": legacy_had_flag,
            "legacy_write_ok": legacy_write_ok,
            "legacy_write_error": legacy_error,
            "legacy_table_has_flag_after": legacy_has_flag,
            "legacy_row_value": legacy_value,
            "legacy_row_backfilled_l1": legacy_backfilled,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis", default=os.environ.get("NEURAL_PODS_REDIS"))
    args = parser.parse_args()
    result = summarise(collect(redis_host=args.redis))
    write_evidence(result, OUT, __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
