"""The storage-facade verdict, checked against observations that say NO.

F3/F4 shipped without the gate checks the master plan promised, and six
defects in the facade survived: the first write stored twice, a read created
tables, predicates were built by string interpolation, an empty namespace
pinned the vector column, and the table listing truncated at ten.

Each one is now a derived field in `summarise()`. The point of these tests
is the direction the old code could not express at all: the measurement used
`assert`, so a facade that got it wrong produced a traceback and NO evidence
file — which the gate reports as "no evidence", a different finding needing
a different fix. Every case below feeds observations from a BROKEN facade
and requires the verdict to say so.
"""
import pytest

from research.benchmark_storage_facade import KV_KEYS, summarise


def _observations(**overrides):
    """One clean run, as collect() would return it."""
    obs = {
        "l1_backend": "in-process stub (no redis host given)",
        "elapsed_s": 9.1,
        "kv_keys": KV_KEYS,
        "kv_rows": KV_KEYS,
        "tables_before": 0,
        "tables_after_read_miss": 0,
        "read_miss_get": None,
        "read_miss_search": 0,
        "read_miss_traces": 0,
        "quoted_key_roundtrips_ok": 2 * KV_KEYS,
        "quoted_key_roundtrips_attempted": 2 * KV_KEYS,
        "write_ms": [13.8] * KV_KEYS,
        "l1_read_ms": [0.05] * KV_KEYS,
        "l2_read_ms": [22.5] * KV_KEYS,
        "l1_entries_before_backfill": 0,
        "l1_entries_after_backfill": KV_KEYS,
        "reput_value_written": {"index": -1},
        "reput_value_read": {"index": -1},
        "rows_after_reput": KV_KEYS,
        "documents": 200,
        "document_rows": 200,
        "document_write_ms": 6.2,
        "top_k_requested": 10,
        "top_k_distinct": 10,
        "namespaces_created": 14,
        "tables_listed": 14,
        "traces_written": 100,
        "trace_rows": 100,
        "quoted_stage_matches": 50,
        "session_affinity": {"reuses": 1, "failovers": 1},
        # A kv table that predates the l1_eligible column: no flag before,
        # the write succeeds, the flag is there afterwards, and the legacy
        # row still reaches L1 on a read.
        "legacy_table_had_flag": False,
        "legacy_write_ok": True,
        "legacy_write_error": None,
        "legacy_table_has_flag_after": True,
        "legacy_row_value": {"index": 0},
        "legacy_row_backfilled_l1": 1,
    }
    obs.update(overrides)
    return obs


def test_a_clean_run_produces_the_verdict_the_gate_accepts():
    report = summarise(_observations())
    kv, l2 = report["kv"], report["l2_lance"]
    assert kv["duplicate_rows"] == 0
    assert kv["stale_read_after_reput"] is False
    assert kv["read_miss_tables_created"] == 0
    assert kv["l1_backfilled_after_l2_hit"] == KV_KEYS
    assert kv["l1_faster_than_l2"] is True
    assert kv["quoted_key_roundtrip"] is True
    assert l2["duplicate_document_rows"] == 0
    assert l2["duplicate_trace_rows"] == 0
    assert l2["tables_listed"] == l2["namespaces_created"]
    assert kv["legacy_table_migrated"] is True
    assert kv["legacy_row_readable"] is True
    assert kv["legacy_row_backfilled_l1"] == 1


def test_the_first_write_stored_twice_shows_up_as_duplicate_rows():
    """Defect 1 of the six. The table holds one row more than there are
    keys, and nothing else about the run looks wrong."""
    report = summarise(_observations(kv_rows=KV_KEYS + 1))
    assert report["kv"]["duplicate_rows"] == 1


def test_a_read_that_created_a_table_is_counted():
    """Defect 2. A miss must not leave anything behind."""
    report = summarise(_observations(tables_after_read_miss=3))
    assert report["kv"]["read_miss_tables_created"] == 3


def test_a_read_miss_that_returned_something_is_not_a_miss():
    """The asserts that used to stand here killed the run, so this case
    never reached the evidence file at all."""
    assert summarise(_observations(read_miss_get={"index": 0})
                     )["kv"]["read_miss_returned_nothing"] is False
    assert summarise(_observations(read_miss_search=2)
                     )["kv"]["read_miss_returned_nothing"] is False
    assert summarise(_observations(read_miss_traces=1)
                     )["kv"]["read_miss_returned_nothing"] is False


def test_a_reput_that_appended_instead_of_replacing_reads_stale():
    """Defect 3. The reader finds the old row first, so the value read back
    is not the value written."""
    report = summarise(_observations(reput_value_read={"index": 0},
                                     rows_after_reput=KV_KEYS + 1))
    assert report["kv"]["stale_read_after_reput"] is True
    assert report["kv"]["rows_after_reput"] == KV_KEYS + 1


def test_a_missing_l1_backfill_is_visible():
    """The facade must place an L2 hit back into L1. Zero back-fill means
    every subsequent read pays the L2 price again."""
    report = summarise(_observations(l1_entries_after_backfill=0))
    assert report["kv"]["l1_backfilled_after_l2_hit"] == 0


def test_a_quoted_key_that_did_not_round_trip_makes_the_field_false():
    """`quoted_key_roundtrip` used to be the literal True in the report, so
    the gate's `is True` check on it could not fail. Now it follows from a
    count, and one apostrophe key lost in a predicate flips it."""
    report = summarise(_observations(quoted_key_roundtrips_ok=2 * KV_KEYS - 1))
    assert report["kv"]["quoted_key_roundtrip"] is False


def test_zero_attempts_is_not_a_pass():
    """A run that made no round-trips at all must not report success by
    virtue of 0 == 0."""
    report = summarise(_observations(quoted_key_roundtrips_ok=0,
                                     quoted_key_roundtrips_attempted=0))
    assert report["kv"]["quoted_key_roundtrip"] is False


def test_l1_slower_than_l2_is_reported_as_such():
    """If this ever comes out False on real hardware, the tier order is not
    doing what the facade claims."""
    report = summarise(_observations(l1_read_ms=[30.0] * KV_KEYS,
                                     l2_read_ms=[22.5] * KV_KEYS))
    assert report["kv"]["l1_faster_than_l2"] is False


def test_the_table_listing_truncating_at_ten_is_visible():
    """Defect 6: table_names() defaults to limit=10, so a facade that does
    not page reports ten of fourteen namespaces."""
    report = summarise(_observations(tables_listed=10))
    l2 = report["l2_lance"]
    assert l2["tables_listed"] == 10
    assert l2["namespaces_created"] == 14
    assert l2["tables_listed"] != l2["namespaces_created"]


def test_duplicate_documents_and_traces_are_counted_separately():
    report = summarise(_observations(document_rows=400, trace_rows=200))
    assert report["l2_lance"]["duplicate_document_rows"] == 200
    assert report["l2_lance"]["duplicate_trace_rows"] == 100


def test_the_l1_backend_is_carried_into_the_report():
    """Whether L1 was a real Redis or the stub decides what the latency
    numbers mean, so it must not be inferable only from context."""
    report = summarise(_observations(l1_backend="redis://10.50.0.121"))
    assert report["l1_backend"] == "redis://10.50.0.121"


def test_a_write_that_failed_against_a_legacy_table_is_reported():
    """lancedb refuses an unknown column outright, so without the migration
    every put() against a grown store raises. The verdict has to show that
    rather than the run simply ending."""
    report = summarise(_observations(
        legacy_write_ok=False,
        legacy_write_error="ValueError(\"Field 'l1_eligible' not found\")",
        legacy_table_has_flag_after=False))
    assert report["kv"]["legacy_table_migrated"] is False
    assert "l1_eligible" in report["kv"]["legacy_write_error"]


def test_a_legacy_row_that_never_reached_l1_again_is_visible():
    """The NULL branch in get(): after the migration the column exists and
    old rows carry NULL, which `dict.get(key, True)` reads as falsy."""
    report = summarise(_observations(legacy_row_backfilled_l1=0))
    assert report["kv"]["legacy_row_backfilled_l1"] == 0
