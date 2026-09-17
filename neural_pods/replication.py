"""Revisioned quorum replication for Pod control-plane records.

The coordinator is backend-neutral. Replicas can be PostgreSQL connections,
remote Pod transports or object-store writers. It requires monotonic revisions,
keeps tombstones and fails closed when the configured quorum is unavailable.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json, threading, time
from typing import Any, Callable, Mapping
import re
try:
    from psycopg.types.json import Jsonb
except ImportError:
    Jsonb = lambda value: value


class PostgresReplica:
    """Durable replica adapter for a psycopg connection.

    Each adapter may point at a separate database/schema/table. Identifier
    names are validated before interpolation; values remain parameterized.
    """
    def __init__(self, connection, table: str):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table): raise ValueError("invalid replica table")
        self.connection, self.table = connection, table
        with connection.cursor() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (item_key text PRIMARY KEY, revision bigint NOT NULL, deleted boolean NOT NULL, record jsonb NOT NULL)")
        connection.commit()

    def put(self, key: str, record: Mapping[str, Any]) -> None:
        with self.connection.cursor() as cur:
            cur.execute(f"INSERT INTO {self.table}(item_key,revision,deleted,record) VALUES(%s,%s,%s,%s) ON CONFLICT(item_key) DO UPDATE SET revision=EXCLUDED.revision,deleted=EXCLUDED.deleted,record=EXCLUDED.record WHERE {self.table}.revision < EXCLUDED.revision", (key, int(record["revision"]), bool(record["deleted"]), Jsonb(dict(record))))
        self.connection.commit()

    def get(self, key: str):
        with self.connection.cursor() as cur:
            cur.execute(f"SELECT record FROM {self.table} WHERE item_key=%s", (key,)); row=cur.fetchone()
        self.connection.commit()
        return row[0] if row else None

    def close(self): self.connection.close()


@dataclass(frozen=True)
class ReplicaResult:
    replica: str
    ok: bool
    revision: int | None = None
    latency_ms: float = 0.0
    error: str | None = None


class QuorumReplicator:
    def __init__(self, replicas: Mapping[str, Any], *, quorum: int | None = None,
                 injector=None, max_retries: int = 0, retry_backoff_s: float = 0.001):
        if not replicas: raise ValueError("at least one replica is required")
        self.replicas = dict(replicas); self.quorum = quorum or (len(self.replicas)//2 + 1)
        if not 1 <= self.quorum <= len(self.replicas): raise ValueError("invalid quorum")
        if max_retries < 0 or retry_backoff_s < 0: raise ValueError("invalid retry policy")
        self.injector, self.max_retries, self.retry_backoff_s = injector, int(max_retries), float(retry_backoff_s)
        self._lock = threading.RLock(); self._revisions: dict[str, int] = {}

    @staticmethod
    def _digest(payload: Mapping[str, Any]) -> str:
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    def write(self, key: str, payload: Mapping[str, Any], *, revision: int | None = None,
              deleted: bool = False) -> dict[str, Any]:
        with self._lock:
            current = self._revisions.get(key, 0); revision = int(revision or current + 1)
            if revision <= current: raise ValueError(f"stale revision: {revision} <= {current}")
            record = {"key": key, "revision": revision, "deleted": bool(deleted),
                      "payload": dict(payload), "digest": self._digest(payload)}
            results = []
            for name, replica in self.replicas.items():
                started = time.perf_counter()
                last = None
                for attempt in range(self.max_retries + 1):
                    try:
                        if self.injector is not None: self.injector.hit(f"replica.{name}.put")
                        replica.put(key, record)
                        results.append(ReplicaResult(name, True, revision, (time.perf_counter()-started)*1000))
                        last = None; break
                    except Exception as exc:
                        last = exc
                        if attempt < self.max_retries:
                            time.sleep(self.retry_backoff_s * (2 ** attempt))
                if last is not None:
                    results.append(ReplicaResult(name, False, error=type(last).__name__, latency_ms=(time.perf_counter()-started)*1000))
            successes = sum(r.ok for r in results)
            if successes < self.quorum: raise RuntimeError(f"quorum unavailable: {successes}/{self.quorum}")
            self._revisions[key] = revision
            return {"key": key, "revision": revision, "quorum": successes, "replicas": [r.__dict__ for r in results]}

    def read(self, key: str) -> dict[str, Any] | None:
        records = []
        for name, replica in self.replicas.items():
            try:
                record = replica.get(key)
                if record is not None: records.append(record)
            except Exception: continue
        if not records: return None
        winner = max(records, key=lambda r: int(r.get("revision", 0)))
        with self._lock: self._revisions[key] = max(self._revisions.get(key, 0), int(winner["revision"]))
        return dict(winner) if not winner.get("deleted") else None

    def tombstone(self, key: str, *, principal: str = "local") -> dict[str, Any]:
        return self.write(key, {"deleted_by": principal}, deleted=True)

    def health(self) -> dict[str, Any]:
        states = {}
        for name, replica in self.replicas.items():
            try: replica.get("__health__"); states[name] = "ok"
            except Exception as exc: states[name] = type(exc).__name__
        return {"replicas": states, "quorum": self.quorum, "available": sum(v == "ok" for v in states.values()) >= self.quorum}
