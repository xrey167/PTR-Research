"""Immutable lineage DAG and a SQLite-serialized, non-streaming commit barrier.

The trusted boundary is this process plus the registry/artifact filesystem.
Hashes detect changed content; they are not signatures or remote attestation.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class InvalidState(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    artifacts: tuple[str, ...]
    principal: str


class Registry:
    def __init__(self, path: str | Path, clock=None):
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        # The registry is written from the threads the rest of the system runs
        # on — household allocations, task-graph nodes, mesh handlers, the
        # adaptive batcher. Without check_same_thread=False any of those
        # raises sqlite3.ProgrammingError on the first event it records.
        # Python's sqlite3 is built in serialized mode (threadsafety == 3), so
        # the connection itself tolerates concurrent statements; what needs
        # serialising is the multi-statement transaction below, so two threads
        # cannot interleave BEGIN/COMMIT on one connection.
        self._lock = threading.RLock()
        self._in_transaction = False
        self.db = sqlite3.connect(str(path), isolation_level=None, timeout=30,
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS nodes(
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
            revoked INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS edges(
            child TEXT REFERENCES nodes(id), parent TEXT REFERENCES nodes(id),
            PRIMARY KEY(child,parent));
        CREATE INDEX IF NOT EXISTS edges_parent ON edges(parent);
        CREATE TABLE IF NOT EXISTS heads(
            knowledge_key TEXT PRIMARY KEY, generation INTEGER NOT NULL,
            node_id TEXT NOT NULL REFERENCES nodes(id));
        CREATE TABLE IF NOT EXISTS events(
            seq INTEGER PRIMARY KEY, action TEXT NOT NULL, payload TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        """One writer at a time. Re-entrant, so a transaction may nest inside
        another on the same thread; the outermost one owns BEGIN/COMMIT."""
        with self._lock:
            outermost = not self._in_transaction
            if outermost:
                self.db.execute("BEGIN IMMEDIATE")
                self._in_transaction = True
            try:
                yield
                if outermost:
                    self.db.execute("COMMIT")
            except BaseException:
                if outermost:
                    self.db.execute("ROLLBACK")
                raise
            finally:
                if outermost:
                    self._in_transaction = False

    def record_event(self, action: str, payload) -> None:
        """Append a provenance event.

        The public write counterpart to events(). Callers outside this module
        used the private _event() because nothing else existed, so the API
        that carries the audit trail was the one marked private.
        """
        with self._lock:
            self.db.execute("INSERT INTO events(action,payload) VALUES(?,?)",
                            (action, canonical(payload)))

    # Internal callers predate record_event(); same function, one name.
    _event = record_event

    def node(self, node_id):
        row = self.db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            raise InvalidState(f"Unknown node: {node_id}")
        return {**dict(row), "payload": json.loads(row["payload"])}

    def _add(self, kind, payload, parents=()):
        parents = sorted(set(parents))
        node_id = f"{kind}:" + digest({"kind": kind, "payload": payload, "parents": parents})
        existing = self.db.execute("SELECT revoked FROM nodes WHERE id=?", (node_id,)).fetchone()
        if existing:
            if existing[0]:
                raise InvalidState("Immutable revoked node cannot be recreated")
            return node_id
        for parent in parents:
            self.node(parent)
        self.db.execute("INSERT INTO nodes(id,kind,payload) VALUES(?,?,?)", (node_id, kind, canonical(payload)))
        self.db.executemany("INSERT INTO edges(child,parent) VALUES(?,?)", [(node_id, p) for p in parents])
        return node_id

    def ancestors(self, node_id):
        self.node(node_id)
        rows = self.db.execute("""WITH RECURSIVE ancestry(id) AS (
            SELECT ? UNION SELECT e.parent FROM edges e JOIN ancestry a ON e.child=a.id)
            SELECT id FROM ancestry""", (node_id,)).fetchall()
        return [self.node(r[0]) for r in rows]

    def roots(self, node_id):
        return sorted(n["id"] for n in self.ancestors(node_id) if n["kind"] == "origin")

    def independent_origins(self, node_ids):
        """Return the deduplicated provenance roots for several artifacts.

        This is a lineage count, not a statistical independence claim: two
        different origin nodes may still describe correlated evidence.  The
        helper is intentionally read-only and is useful when scoring evidence
        without counting repeated derived artifacts as new sources.
        """
        roots = set()
        for node_id in node_ids:
            roots.update(self.roots(node_id))
        return sorted(roots)

    def _valid(self, node_id, principal):
        for node in self.ancestors(node_id):
            if node["revoked"]:
                raise InvalidState(f"Revoked ancestor: {node['id']}")
            payload = node["payload"]
            acl = payload.get("acl", ["*"])
            if "*" not in acl and principal not in acl:
                raise InvalidState("Access denied by lineage ACL")
            if node["kind"] == "knowledge":
                lifecycle = payload.get("lifecycle", {})
                now = self.clock()
                if lifecycle.get("valid_from") and now < datetime.fromisoformat(lifecycle["valid_from"]):
                    raise InvalidState("Knowledge is not yet valid")
                if lifecycle.get("valid_until") and now >= datetime.fromisoformat(lifecycle["valid_until"]):
                    raise InvalidState("Knowledge has expired")
                row = self.db.execute("SELECT node_id FROM heads WHERE knowledge_key=?", (payload["knowledge_key"],)).fetchone()
                if row is None or row[0] != node["id"]:
                    raise InvalidState("Stale knowledge generation")

    def origin(self, namespace, record_id, version, content, acl=("*",)):
        # Structured encoding avoids delimiter/catenation ambiguity.
        payload = dict(namespace=namespace, record_id=record_id, version=version,
                       content_hash=digest(content), acl=sorted(set(acl)))
        with self.transaction():
            return self._add("origin", payload)

    def publish(self, knowledge_key, semantic, parents, principal="local", acl=("*",), lifecycle=None):
        parents = tuple(parents)
        if not parents:
            raise InvalidState("Knowledge requires source lineage")
        if lifecycle is not None:
            lifecycle = dict(lifecycle)
            for field in ("valid_from", "valid_until"):
                if lifecycle.get(field):
                    parsed = datetime.fromisoformat(lifecycle[field])
                    if parsed.tzinfo is None:
                        raise ValueError("Validity timestamps must include a timezone")
                    lifecycle[field] = parsed.astimezone(timezone.utc).isoformat()
            if lifecycle.get("valid_from") and lifecycle.get("valid_until"):
                if lifecycle["valid_from"] >= lifecycle["valid_until"]:
                    raise ValueError("Invalid validity interval")
        with self.transaction():
            for parent in parents:
                self._valid(parent, principal)
            previous = self.db.execute("SELECT generation FROM heads WHERE knowledge_key=?", (knowledge_key,)).fetchone()
            generation = 1 if previous is None else previous[0] + 1
            payload = dict(knowledge_key=knowledge_key, generation=generation,
                           semantic=semantic, acl=sorted(set(acl)))
            if lifecycle is not None:
                lifecycle["supersedes"] = self.head(knowledge_key) if previous else None
                payload["lifecycle"] = lifecycle
            node_id = self._add("knowledge", payload, parents)
            self.db.execute("INSERT INTO heads VALUES(?,?,?) ON CONFLICT(knowledge_key) DO UPDATE SET generation=excluded.generation,node_id=excluded.node_id", (knowledge_key, generation, node_id))
            self._event("publish", {"node": node_id, "knowledge": knowledge_key, "generation": generation})
            return node_id

    def _artifact(self, kind, payload, parents, principal):
        if kind not in {"lora", "model", "vector", "text", "jspace", "cache", "answer", "router",
                        "embedding", "ranker", "speculator", "runtime", "policy", "quantization", "vision"}:
            raise ValueError("Unsupported artifact kind")
        parents = tuple(parents)
        if not parents:
            raise InvalidState("No artifact without lineage")
        knowledge = {}
        origins = set()
        for parent in parents:
            self._valid(parent, principal)
            for n in self.ancestors(parent):
                if n["kind"] == "origin":
                    origins.add(n["id"])
                if n["kind"] == "knowledge":
                    p = n["payload"]
                    knowledge[n["id"]] = {"knowledge_key": p["knowledge_key"], "generation": p["generation"]}
        if not knowledge or not origins:
            raise InvalidState("Artifact requires canonical knowledge and provenance roots")
        header = dict(payload=payload, origin_keys=sorted(origins), generations=knowledge,
                      parent_artifact_keys=sorted(set(parents)), derivation_hash=digest(payload))
        return self._add(kind, header, parents)

    def artifact(self, kind, payload, parents, principal="local"):
        with self.transaction():
            return self._artifact(kind, payload, parents, principal)

    def snapshot(self, artifacts: Iterable[str], principal="local"):
        ids = tuple(sorted(set(artifacts)))
        if not ids:
            raise InvalidState("No valid neural evidence")
        with self.transaction():
            for node_id in ids:
                self._valid(node_id, principal)
        return Snapshot(ids, principal)

    def commit(self, snapshot: Snapshot, answer: str):
        # Check and durable materialization share the same database write lock.
        with self.transaction():
            for node_id in snapshot.artifacts:
                self._valid(node_id, snapshot.principal)
            answer_id = self._artifact("answer", {"text": answer}, snapshot.artifacts, snapshot.principal)
            self._event("commit", {"answer": answer_id})
            return {"answer_id": answer_id, "text": answer, "dependencies": list(snapshot.artifacts)}

    def revoke(self, node_id):
        with self.transaction():
            self.node(node_id)
            affected = [r[0] for r in self.db.execute("""WITH RECURSIVE closure(id) AS (
                SELECT ? UNION SELECT e.child FROM edges e JOIN closure c ON e.parent=c.id)
                SELECT id FROM closure""", (node_id,))]
            self.db.executemany("UPDATE nodes SET revoked=1 WHERE id=?", [(a,) for a in affected])
            self._event("revoke", {"root": node_id, "affected": affected})
            return sorted(affected)

    def events(self, *, action: str | None = None, limit: int = 1000) -> list[dict]:
        """List provenance events, newest first; optionally filter by action."""
        if action is not None:
            rows = self.db.execute(
                "SELECT seq, action, payload FROM events WHERE action=? "
                "ORDER BY seq DESC LIMIT ?", (action, limit)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT seq, action, payload FROM events ORDER BY seq DESC LIMIT ?",
                (limit,)).fetchall()
        return [{"seq": r[0], "action": r[1], "payload": json.loads(r[2])}
                for r in rows]

    def head(self, knowledge_key):
        row = self.db.execute("SELECT node_id FROM heads WHERE knowledge_key=?", (knowledge_key,)).fetchone()
        if not row:
            raise InvalidState("Unknown knowledge key")
        return row[0]


def hash_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise InvalidState("Artifact directory is missing")
    result = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            if p.is_symlink():
                raise InvalidState("Symlink in artifact directory")
            with p.open("rb") as handle:
                result[p.relative_to(directory).as_posix()] = hashlib.file_digest(handle, "sha256").hexdigest()
    if not result:
        raise InvalidState("Artifact directory is empty")
    return result


def verify_files(directory, expected):
    if hash_files(directory) != expected:
        raise InvalidState("Artifact files differ from registered hashes")
