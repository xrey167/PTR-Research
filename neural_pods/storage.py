"""PodStorage: the single storage API pods talk to.

Tiers (Storage-Architecture-Design):
  L0 pod-local process state (crossbeam-pattern structures, ephemeral)
  L1 Redis shared-hot (mesh cache tier, existing)
  L2 LanceDB warm columnar (vectors, documents, traces — versioned)
  L3 cold snapshots (SnapshotStore -> SSD/object)

put() writes L1 (TTL) and L2 (persistent); get() reads L1 first, then L2 and
back-fills L1. search_vectors/traces/traces_query go straight to Lance.
Session affinity (kv_session) implements the Mooncake pattern: a session
sticks to the replica holding its warm KV cache, with failover counted.

Two invariants this module got wrong before and now enforces explicitly:
  * a read NEVER creates a table — `_open` returns None instead, so a miss
    cannot leave a placeholder row behind and cannot pin a vector column to
    a dummy dimension;
  * a first write creates the table WITH its rows and stops there — the old
    create-then-add path wrote every first batch twice, which made top-k
    searches return the same document more than once.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _sql_literal(value: Any) -> str:
    """Single-quoted SQL string literal for Lance filters; ' is doubled.

    Keys arrive from callers (case ids, namespaces, principals) and used to
    be interpolated raw, so a single apostrophe broke the predicate.
    """
    return "'" + str(value).replace("'", "''") + "'"


class SessionAffinity:
    """Mooncake-pattern session -> replica mapping with failover metric."""

    def __init__(self):
        self.mapping: dict[str, str] = {}
        self.failovers = 0
        self.reuses = 0

    def bind(self, session_id: str, replica: str) -> str:
        previous = self.mapping.get(session_id)
        if previous is None:
            self.mapping[session_id] = replica
        elif previous != replica:
            self.failovers += 1  # warm KV lost, prefix must be rebuilt
            self.mapping[session_id] = replica
        else:
            self.reuses += 1  # warm KV cache reused on the same replica
        return self.mapping[session_id]

    def stats(self) -> dict[str, Any]:
        return {"sessions": len(self.mapping), "failovers": self.failovers,
                "reuses": self.reuses}


class PodStorage:
    def __init__(self, *, redis_client: Any = None,
                 lance_dir: str | Path = "storage/lance",
                 ttl_s: int = 600, principal: str = "local"):
        self.redis = redis_client
        self.principal = principal
        self.ttl_s = ttl_s
        self.lance_dir = str(lance_dir)
        self._lance = None
        self.session_affinity = SessionAffinity()

    # --- L2 LanceDB (lazy) ------------------------------------------------
    def _lance_db(self):
        if self._lance is None:
            import lancedb
            self._lance = lancedb.connect(self.lance_dir)
        return self._lance

    def _tables(self) -> list[str]:
        """All table names, paged.

        list_tables() replaced table_names() and returns a paginated response
        object rather than a list. table_names() also defaults to limit=10,
        which silently truncates once a pod holds more than ten namespaces —
        a membership test against a truncated list would then try to create a
        table that already exists.
        """
        db = self._lance_db()
        lister = getattr(db, "list_tables", None)
        if lister is None:
            return list(db.table_names(limit=100_000))
        names: list[str] = []
        token = None
        while True:
            page = lister(page_token=token) if token else lister()
            names.extend(getattr(page, "tables", page) or [])
            token = getattr(page, "page_token", None)
            if not token:
                return names

    def _open(self, name: str):
        """Existing table or None. Reads go through here: a lookup must not
        create anything."""
        if name not in self._tables():
            return None
        return self._lance_db().open_table(name)

    def _evolve_schema(self, table, rows: list[dict[str, Any]]):
        """Add columns `rows` carry that the table does not have yet.

        Needed because `merge_insert` casts the incoming rows HARD against the
        table schema and refuses an unknown column outright. Measured against
        lancedb 0.39.0:

            ValueError: Field 'l1_eligible' not found in target schema

        `allow_subschema` only permits FEWER columns, never more, and the
        merge builder has no evolution flag. So a table created before a
        column existed does not lose the value quietly — every single write
        against it raises. Without this, deploying the `l1_eligible` column
        would have broken `put()` on every store that already held data.

        The new column is nullable and existing rows get NULL: a metadata
        operation that does not rewrite a grown table. Reading NULL means
        "no opinion" — see `get()`, which has to distinguish that from False.

        Idempotent: nothing missing, nothing written, so a fresh table and
        the settled state are both no-ops.
        """
        import pyarrow as pa

        have = set(table.schema.names)
        incoming = pa.Table.from_pylist(rows).schema
        missing = [incoming.field(name) for name in incoming.names
                   if name not in have]
        if not missing:
            return table
        table.add_columns(pa.schema([field.with_nullable(True)
                                     for field in missing]))
        return table

    def _write(self, name: str, rows: list[dict[str, Any]],
               *, keys: list[str] | None = None, evolve: bool = False):
        """Create-with-rows on the first write, upsert (keys) or append after.

        The create branch deliberately does not add() afterwards — the table
        already holds `rows`.

        `evolve` is opt-in per call, not automatic for every table. `docs_*`
        and `traces` take rows from arbitrary callers; evolving there would
        turn a typo in a row dict into a silent schema change on a production
        table. `kv` is the one table whose row shape this module controls.
        """
        db = self._lance_db()
        if name not in self._tables():
            return db.create_table(name, data=rows)
        table = db.open_table(name)
        if evolve:
            table = self._evolve_schema(table, rows)
        if keys:
            table.merge_insert(keys).when_matched_update_all() \
                 .when_not_matched_insert_all().execute(rows)
        else:
            table.add(rows)
        return table

    # --- KV tiering (L1 -> L2) ---------------------------------------------
    def _l1_key(self, key: str) -> str:
        return "np:storage:" + self.principal + ":" + hashlib.sha256(
            json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def _l1_put(self, key: str, encoded: str) -> None:
        if self.redis is not None:
            self.redis.set(self._l1_key(key), encoded, ex=self.ttl_s)

    def put(self, key: str, value: Any, *, l1: bool = True) -> None:
        """Write to L2, and to L1 unless the caller opts out.

        `l1=False` is a property of the VALUE, not of one call, so it is
        stored with the row. Otherwise the first get() would put the value
        into the shared hot tier the writer just asked to keep it out of.
        """
        encoded = json.dumps(value, default=str)
        if l1:
            self._l1_put(key, encoded)
        # (key, principal) is the identity: a re-put replaces, it does not
        # append a second row that a later get() might read instead.
        self._write("kv", [{"key": key, "value": encoded,
                            "principal": self.principal, "ts": time.time(),
                            "l1_eligible": bool(l1)}],
                    keys=["key", "principal"], evolve=True)

    def get(self, key: str) -> Any | None:
        if self.redis is not None:
            raw = self.redis.get(self._l1_key(key))
            if raw is not None:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                return json.loads(raw)
        table = self._open("kv")
        if table is None:
            return None
        rows = table.search().where(
            f"key = {_sql_literal(key)} AND "
            f"principal = {_sql_literal(self.principal)}",
            prefilter=True).limit(1).to_list()
        if not rows:
            return None
        encoded = rows[0]["value"]
        # Warm L1 back up after an L2 hit - but only for values that were
        # allowed into L1 in the first place.
        #
        # "No opinion" has TWO shapes, and `dict.get(key, default)` only
        # catches the first: the key is absent (a table not yet migrated), or
        # the key is present with NULL (a row written before the column, in a
        # table that has since been migrated). `to_list()` returns every
        # column of the table, so after the migration the key is always there
        # and the default never applies — None is falsy, and the back-fill
        # this comment promises would have been skipped in silence.
        flag = rows[0].get("l1_eligible")
        if flag is None or bool(flag):
            self._l1_put(key, encoded)
        return json.loads(encoded)

    # --- Lance vectors/documents --------------------------------------------
    def add_documents(self, namespace: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._write(f"docs_{namespace}", rows)

    def search_documents(self, namespace: str, query_vector: list[float],
                         top_k: int = 10) -> list[dict[str, Any]]:
        table = self._open(f"docs_{namespace}")
        if table is None:
            return []  # unknown namespace: empty result, not a dummy schema
        return table.search(query_vector).limit(top_k).to_list()

    # --- Traces ---------------------------------------------------------------
    def write_traces(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self._write("traces", records)

    def query_traces(self, *, stage: str | None = None,
                     limit: int = 100) -> list[dict[str, Any]]:
        table = self._open("traces")
        if table is None:
            return []
        query = table.search()
        if stage is not None:
            query = query.where(f"stage = {_sql_literal(stage)}", prefilter=True)
        return query.limit(limit).to_list()

    # --- Session affinity (Mooncake pattern) -----------------------------------
    def kv_session(self, session_id: str, replica: str) -> str:
        return self.session_affinity.bind(session_id, replica)

    def stats(self) -> dict[str, Any]:
        return {"principal": self.principal, "lance_dir": self.lance_dir,
                "session_affinity": self.session_affinity.stats()}
