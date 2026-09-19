"""PodStorage: the single storage API pods talk to.

Tiers (Storage-Architecture-Design):
  L0 pod-local process state (crossbeam-pattern structures, ephemeral)
  L1 Redis shared-hot (mesh cache tier, existing)
  L2 LanceDB warm columnar (vectors, documents, traces — versioned)
  L3 cold snapshots (SnapshotStore -> SSD/object)

put() writes L1 (TTL) and L2 (persistent); get() reads L1 first, then L2.
search_vectors/traces/traces_query go straight to Lance. Session affinity
(kv_session) implements the Mooncake pattern: a session sticks to the
replica holding its warm KV cache, with failover counted.
"""
from __future__ import annotations
import json
import time
from typing import Any

import hashlib


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
    def __init__(self, *, redis_client: Any = None, lance_dir: str | Path = "storage/lance",
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

    def _ensure_table(self, name: str, schema_rows: list[dict]):
        db = self._lance_db()
        if name not in db.table_names():
            db.create_table(name, data=schema_rows)
        return db.open_table(name)

    # --- KV tiering (L1 -> L2) ---------------------------------------------
    def _l1_key(self, key: str) -> str:
        return "np:storage:" + self.principal + ":" + hashlib.sha256(
            json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def put(self, key: str, value: Any, *, l1: bool = True) -> None:
        if self.redis is not None and l1:
            self.redis.set(self._l1_key(key), json.dumps(value, default=str),
                           ex=self.ttl_s)
        table = self._ensure_table("kv", [{"key": key, "value": json.dumps(
            value, default=str), "principal": self.principal, "ts": time.time()}])
        table.add([{"key": key, "value": json.dumps(value, default=str),
                    "principal": self.principal, "ts": time.time()}])

    def get(self, key: str) -> Any | None:
        if self.redis is not None:
            raw = self.redis.get(self._l1_key(key))
            if raw is not None:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                return json.loads(raw)
        table = self._ensure_table("kv", [{"key": key, "value": "null",
                                           "principal": self.principal,
                                           "ts": time.time()}])
        rows = table.search().where(
            f"key = '{key}' AND principal = '{self.principal}'",
            prefilter=True).limit(1).to_list()
        return json.loads(rows[0]["value"]) if rows else None

    # --- Lance vectors/documents --------------------------------------------
    def add_documents(self, namespace: str, rows: list[dict[str, Any]]) -> None:
        table = self._ensure_table(f"docs_{namespace}", rows)
        table.add(rows)

    def search_documents(self, namespace: str, query_vector: list[float],
                         top_k: int = 10) -> list[dict[str, Any]]:
        table = self._ensure_table(f"docs_{namespace}", [
            {"text": "", "vector": [0.0], "key": ""}])
        return table.search(query_vector).limit(top_k).to_list()

    # --- Traces ---------------------------------------------------------------
    def write_traces(self, records: list[dict[str, Any]]) -> None:
        self._ensure_table("traces", records)
        self._lance_db().open_table("traces").add(records)

    def query_traces(self, *, stage: str | None = None,
                     limit: int = 100) -> list[dict[str, Any]]:
        table = self._ensure_table("traces", [{
            "trace_id": "", "case": "", "stage": "", "latency_ms": 0.0}])
        query = table.search()
        if stage is not None:
            query = query.where(f"stage = '{stage}'", prefilter=True)
        return query.limit(limit).to_list()

    # --- Session affinity (Mooncake pattern) -----------------------------------
    def kv_session(self, session_id: str, replica: str) -> str:
        return self.session_affinity.bind(session_id, replica)

    def stats(self) -> dict[str, Any]:
        return {"principal": self.principal, "lance_dir": self.lance_dir,
                "session_affinity": self.session_affinity.stats()}
