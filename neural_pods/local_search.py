"""Self-hosted, Turbopuffer-inspired search for semantic Pods.

The backend is deliberately a small local primitive: SQLite is the durable
namespace store and readers build an in-process snapshot on demand.  It has no
Turbopuffer dependency or network service.  Registry validation remains the
source of truth; this module only returns candidates and applies hard payload
filters before scoring.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Mapping

from .ranking import TensorRankProfile


_WORD = re.compile(r"[\w-]+", re.UNICODE)
_UNSET = object()


def _tokens(text: str) -> list[str]:
    return [x.casefold() for x in _WORD.findall(text or "")]


def _cosine(a: Iterable[float] | None, b: Iterable[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    aa, bb = list(a), list(b)
    if len(aa) != len(bb) or not aa:
        return 0.0
    dot = sum(x * y for x, y in zip(aa, bb))
    na = math.sqrt(sum(x * x for x in aa))
    nb = math.sqrt(sum(y * y for y in bb))
    return dot / (na * nb) if na and nb else 0.0


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SearchHit:
    key: str
    score: float
    vector_score: float
    lexical_score: float
    regex_score: float
    text: str
    metadata: dict[str, Any]


class LocalSearchBackend:
    """Durable namespaces with stateless, concurrent readers.

    Branches are copy-on-write: a branch resolves its own rows first and then
    falls back to its parent.  A reader never mutates the durable store and may
    safely be called from multiple threads.  ``pin`` only controls the local
    snapshot cache; deleting it simulates scale-to-zero/cold storage.
    """

    def __init__(self, path: str | Path, *, embedder: Any | None = None):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS branches(
            namespace TEXT NOT NULL, branch TEXT NOT NULL,
            parent TEXT, PRIMARY KEY(namespace, branch));
        CREATE TABLE IF NOT EXISTS items(
            namespace TEXT NOT NULL, branch TEXT NOT NULL, item_key TEXT NOT NULL,
            text TEXT NOT NULL, vector TEXT, metadata TEXT NOT NULL,
            deleted INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(namespace, branch, item_key));
        CREATE INDEX IF NOT EXISTS item_lookup ON items(namespace, branch, item_key);
        CREATE TABLE IF NOT EXISTS namespace_metadata(
            namespace TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            last_write_at TEXT, updated_at TEXT NOT NULL,
            read_only INTEGER NOT NULL DEFAULT 0, pinning TEXT,
            schema_json TEXT, revision INTEGER NOT NULL DEFAULT 0);
        """)
        self.db.execute("INSERT OR IGNORE INTO branches VALUES(?,?,NULL)", ("default", "main"))
        if "schema_json" not in {row[1] for row in self.db.execute("PRAGMA table_info(namespace_metadata)")}: 
            self.db.execute("ALTER TABLE namespace_metadata ADD COLUMN schema_json TEXT")
        if "revision" not in {row[1] for row in self.db.execute("PRAGMA table_info(namespace_metadata)")}: 
            self.db.execute("ALTER TABLE namespace_metadata ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute("INSERT OR IGNORE INTO namespace_metadata(namespace,created_at,updated_at) VALUES(?,?,?)", ("default", now, now))
        self.db.commit()
        self._lock = RLock()
        self._pinned: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        self._filter_cache: dict[tuple[str,str,str,str], frozenset[str]] = {}
        self._postings_cache: dict[tuple[str,str], dict[str,frozenset[str]]] = {}
        self._block_postings: dict[tuple[str,str,str], object] = {}
        # Optional local callable: text -> finite vector. No hosted embedding
        # provider is contacted; callers may bind Vela/SentenceTransformer.
        self.embedder = embedder
        self.filter_cache_hits = self.filter_cache_misses = 0

    def build_block_postings(self, namespace: str, branch: str = "main", field: str = "permissions", block_size: int = 256):
        """Build a fixed-size posting index for an array metadata field."""
        from .block_postings import BlockPostings
        snapshot = self._snapshot(namespace, branch); terms = {}
        for item in snapshot.values():
            if item["deleted"]: continue
            for value in item["metadata"].get(field, []): terms.setdefault(str(value), []).append(int(item["key"]) if str(item["key"]).isdigit() else abs(hash(item["key"])) % (2**63))
        index = BlockPostings(block_size); [index.add(term, ids) for term, ids in terms.items()]
        self._block_postings[(namespace, branch, field)] = index
        return index.stats()

    def build_vector_index(self, namespace: str, branch: str = "main", path: str | Path | None = None,
                           *, approximate: bool = True, ef_search: int = 64):
        """Materialize a persistent dense index for a namespace snapshot.

        The index is an acceleration layer only: ACL, generation and other
        metadata gates remain in ``search``.  Callers can use its returned
        object for unfiltered ANN fan-out and retain exact search for audits.
        """
        from .vector_index import HNSWVectorIndex, PersistentVectorIndex
        snapshot = self._snapshot(namespace, branch)
        rows = [(x["key"], x["vector"]) for x in snapshot.values() if not x["deleted"] and x["vector"] is not None]
        if not rows: return {"vectors": 0, "dimension": None, "storage": "empty", "exact": True}
        dimension = len(rows[0][1]); target = path or (Path(self.path).with_suffix(f".{namespace}.{branch}.vectors"))
        if approximate:
            try: index = HNSWVectorIndex(target, dimension=dimension, ef_search=ef_search)
            except RuntimeError: index = PersistentVectorIndex(target, dimension=dimension)
        else: index = PersistentVectorIndex(target, dimension=dimension)
        if hasattr(index, "bulk_upsert"): index.bulk_upsert(rows)
        else:
            for key, vector in rows: index.upsert(key, vector)
        index.save(); return index.stats()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def _ensure_branch(self, namespace: str, branch: str) -> None:
        row = self.db.execute("SELECT 1 FROM branches WHERE namespace=? AND branch=?", (namespace, branch)).fetchone()
        if row is None:
            raise KeyError(f"unknown namespace/branch: {namespace}/{branch}")

    def create_namespace(self, namespace: str, branch: str = "main") -> None:
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO branches VALUES(?,?,NULL)", (namespace, branch))
            now = datetime.now(timezone.utc).isoformat()
            self.db.execute("INSERT OR IGNORE INTO namespace_metadata(namespace,created_at,updated_at) VALUES(?,?,?)", (namespace, now, now))
            self.db.commit()

    def namespace_metadata(self, namespace: str) -> dict[str, Any]:
        """Return turbopuffer-style namespace state for observability."""
        with self._lock:
            row = self.db.execute("SELECT * FROM namespace_metadata WHERE namespace=?", (namespace,)).fetchone()
            if row is None: raise KeyError(f"unknown namespace: {namespace}")
            branches = self.db.execute("SELECT branch,parent FROM branches WHERE namespace=?", (namespace,)).fetchall()
            count = self.db.execute("SELECT COUNT(*) FROM items WHERE namespace=? AND deleted=0", (namespace,)).fetchone()[0]
            out = {"approx_row_count": int(count), "created_at": row["created_at"],
                   "last_write_at": row["last_write_at"], "updated_at": row["updated_at"],
                   "revision": int(row["revision"] or 0),
                   "index": {"status": "up-to-date", "unindexed_bytes": 0},
                   "branches": [{"branch": x["branch"], "parent": x["parent"]} for x in branches]}
            if row["schema_json"]: out["schema"] = json.loads(row["schema_json"])
            if row["read_only"]: out["read_only"] = True
            if row["pinning"]: out["pinning"] = json.loads(row["pinning"])
            return out

    def configure_schema(self, namespace: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Configure local typed/index hints, including a native embedder."""
        with self._lock:
            if self.db.execute("SELECT 1 FROM namespace_metadata WHERE namespace=?", (namespace,)).fetchone() is None:
                raise KeyError(f"unknown namespace: {namespace}")
            now = datetime.now(timezone.utc).isoformat()
            self.db.execute("UPDATE namespace_metadata SET schema_json=?,updated_at=?,revision=revision+1 WHERE namespace=?", (_json(dict(schema)), now, namespace)); self.db.commit()
            self._invalidate_pins(namespace, "*")
            return self.namespace_metadata(namespace)

    def _embedding_config(self, namespace: str) -> Mapping[str, Any]:
        row = self.db.execute("SELECT schema_json FROM namespace_metadata WHERE namespace=?", (namespace,)).fetchone()
        schema = json.loads(row[0]) if row and row[0] else {}
        text_schema = schema.get("text", {}) if isinstance(schema, Mapping) else {}
        embed = text_schema.get("embed") if isinstance(text_schema, Mapping) else None
        return embed if isinstance(embed, Mapping) else ({"model": embed} if embed else {})

    def reembed_namespace(self, namespace: str, *, branch: str = "main",
                          embedder: Any | None = None) -> dict[str, Any]:
        """Re-ingest all source text with the current/local embedding model.

        This is explicit by design: changing schema metadata never silently
        rewrites vectors. Each rewritten row gets an incremented revision and
        the namespace timestamp changes atomically from the reader's point of
        view.
        """
        fn = embedder or self.embedder
        if fn is None: raise RuntimeError("re-embedding requires a local embedder")
        with self._lock:
            self._ensure_branch(namespace, branch)
            meta_row = self.db.execute("SELECT read_only FROM namespace_metadata WHERE namespace=?", (namespace,)).fetchone()
            if meta_row and meta_row[0]: raise PermissionError(f"namespace is read-only: {namespace}")
            rows = [x for x in self._rows(namespace, branch).values() if not x["deleted"]]
            for item in rows:
                vector = list(fn(item["text"]))
                expected_dim = self._embedding_config(namespace).get("dims")
                if expected_dim is not None and len(vector) != int(expected_dim):
                    raise ValueError(f"embedding dimension mismatch: expected {expected_dim}, got {len(vector)}")
                metadata = dict(item["metadata"]); metadata["revision"] = int(metadata.get("revision", 1)) + 1
                metadata["embedding_model"] = self._embedding_config(namespace).get("model", "local")
                metadata["embedding_dimension"] = len(vector)
                self.db.execute("""INSERT INTO items(namespace,branch,item_key,text,vector,metadata,deleted)
                    VALUES(?,?,?,?,?,?,0) ON CONFLICT(namespace,branch,item_key) DO UPDATE SET
                    text=excluded.text, vector=excluded.vector, metadata=excluded.metadata, deleted=0""",
                    (namespace, branch, item["key"], item["text"], _json(vector), _json(metadata)))
            now = datetime.now(timezone.utc).isoformat()
            self.db.execute("UPDATE namespace_metadata SET last_write_at=?,updated_at=?,revision=revision+1 WHERE namespace=?", (now, now, namespace)); self.db.commit()
            self._invalidate_pins(namespace, branch)
            return {"namespace": namespace, "branch": branch, "rows_reembedded": len(rows), "updated_at": now}

    def update_namespace_metadata(self, namespace: str, *, read_only: bool | None = None,
                                  pinning: Mapping[str, Any] | None | object = _UNSET) -> dict[str, Any]:
        """Update local pinning/read-only controls, mirroring the metadata API."""
        with self._lock:
            if self.db.execute("SELECT 1 FROM namespace_metadata WHERE namespace=?", (namespace,)).fetchone() is None:
                raise KeyError(f"unknown namespace: {namespace}")
            sets, values = [], []
            if read_only is not None: sets.append("read_only=?"); values.append(int(read_only))
            if pinning is not _UNSET: sets.append("pinning=?"); values.append(None if pinning is None else _json(dict(pinning)))
            if sets:
                sets.append("updated_at=?"); values.append(datetime.now(timezone.utc).isoformat()); values.append(namespace)
                self.db.execute(f"UPDATE namespace_metadata SET {', '.join(sets)},revision=revision+1 WHERE namespace=?", values); self.db.commit()
                self._invalidate_pins(namespace, "*")
            return self.namespace_metadata(namespace)

    def branch(self, namespace: str, source: str = "main", target: str | None = None) -> str:
        target = target or f"branch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        with self._lock:
            self._ensure_branch(namespace, source)
            if self.db.execute("SELECT 1 FROM branches WHERE namespace=? AND branch=?", (namespace, target)).fetchone():
                raise ValueError(f"branch already exists: {target}")
            self.db.execute("INSERT INTO branches VALUES(?,?,?)", (namespace, target, source))
            self.db.commit()
        return target

    def _rows(self, namespace: str, branch: str) -> dict[str, dict[str, Any]]:
        with self._lock:
            self._ensure_branch(namespace, branch)
            chain: list[str] = []
            current = branch
            seen: set[str] = set()
            while current is not None:
                if current in seen:
                    raise RuntimeError("branch parent cycle detected")
                seen.add(current)
                chain.append(current)
                row = self.db.execute("SELECT parent FROM branches WHERE namespace=? AND branch=?", (namespace, current)).fetchone()
                current = row[0] if row else None
            result: dict[str, dict[str, Any]] = {}
            for level in chain:  # child wins over parent
                for row in self.db.execute("SELECT * FROM items WHERE namespace=? AND branch=?", (namespace, level)):
                    if row["item_key"] in result:
                        continue
                    result[row["item_key"]] = {
                        "key": row["item_key"], "text": row["text"],
                        "vector": json.loads(row["vector"]) if row["vector"] else None,
                        "metadata": json.loads(row["metadata"]), "deleted": bool(row["deleted"]),
                    }
            return result

    def _invalidate_pins(self, namespace: str, changed_branch: str) -> None:
        """Invalidate a branch and every descendant snapshot after a write."""
        # Any write can affect descendants through copy-on-write inheritance;
        # clear all derived indexes for this namespace for correctness.
        self._filter_cache = {k:v for k,v in self._filter_cache.items() if k[0] != namespace}
        self._postings_cache = {k:v for k,v in self._postings_cache.items() if k[0] != namespace}
        self._block_postings = {k:v for k,v in self._block_postings.items() if k[0] != namespace}
        for key in list(self._pinned):
            ns, candidate = key
            if ns != namespace:
                continue
            current = candidate
            while current is not None:
                if current == changed_branch:
                    self._pinned.pop(key, None)
                    break
                row = self.db.execute("SELECT parent FROM branches WHERE namespace=? AND branch=?", (namespace, current)).fetchone()
                current = row[0] if row else None

    def upsert(self, namespace: str, key: str, *, text: str = "", vector: Iterable[float] | None = None,
               metadata: Mapping[str, Any] | None = None, branch: str = "main") -> None:
        metadata = dict(metadata or {})
        metadata.setdefault("revision", 1)
        if vector is None and self.embedder is not None:
            embedding = self._embedding_config(namespace)
            if embedding:
                vector = self.embedder(text)
                metadata["embedding_model"] = embedding.get("model", "local")
                metadata["embedding_dimension"] = len(vector)
        if vector is not None:
            vector = list(vector)
            if not vector or not all(math.isfinite(float(value)) for value in vector):
                raise ValueError("vector must contain finite values")
            expected_dim = self._embedding_config(namespace).get("dims")
            if expected_dim is not None and len(vector) != int(expected_dim):
                raise ValueError(f"embedding dimension mismatch: expected {expected_dim}, got {len(vector)}")
        with self._lock:
            self._ensure_branch(namespace, branch)
            meta_row = self.db.execute("SELECT read_only FROM namespace_metadata WHERE namespace=?", (namespace,)).fetchone()
            if meta_row and meta_row[0]: raise PermissionError(f"namespace is read-only: {namespace}")
            self.db.execute("""INSERT INTO items(namespace,branch,item_key,text,vector,metadata,deleted)
                VALUES(?,?,?,?,?,?,0) ON CONFLICT(namespace,branch,item_key) DO UPDATE SET
                text=excluded.text, vector=excluded.vector, metadata=excluded.metadata, deleted=0""",
                (namespace, branch, key, text, _json(vector) if vector is not None else None, _json(metadata)))
            self.db.commit()
            now = datetime.now(timezone.utc).isoformat(); self.db.execute("UPDATE namespace_metadata SET last_write_at=?,updated_at=?,revision=revision+1 WHERE namespace=?", (now, now, namespace)); self.db.commit()
            self._invalidate_pins(namespace, branch)

    def partial_update(self, namespace: str, key: str, *, branch: str = "main",
                       metadata: Mapping[str, Any] | None = None,
                       text: str | object = _UNSET,
                       vector: Iterable[float] | None | object = _UNSET,
                       expected_revision: int | None = None) -> int:
        """Apply a realtime field update without replacing the whole Pod.

        Updates are copy-on-write at branch level and increment a per-record
        revision.  ``expected_revision`` provides test-and-set semantics so a
        stale updater cannot overwrite a newer generation projection.
        """
        with self._lock:
            self._ensure_branch(namespace, branch)
            meta_row = self.db.execute("SELECT read_only FROM namespace_metadata WHERE namespace=?", (namespace,)).fetchone()
            if meta_row and meta_row[0]: raise PermissionError(f"namespace is read-only: {namespace}")
            current = self._rows(namespace, branch).get(key)
            if current is None or current["deleted"]:
                raise KeyError(f"unknown item: {namespace}/{branch}/{key}")
            current_revision = int(current["metadata"].get("revision", 1))
            if expected_revision is not None and expected_revision != current_revision:
                raise ValueError(f"revision conflict: expected {expected_revision}, current {current_revision}")
            merged = dict(current["metadata"])
            merged.update(dict(metadata or {}))
            merged["revision"] = current_revision + 1
            new_text = current["text"] if text is _UNSET else str(text)
            new_vector = current["vector"] if vector is _UNSET else vector
            if new_vector is not None:
                new_vector = list(new_vector)
                if not new_vector or not all(math.isfinite(float(value)) for value in new_vector):
                    raise ValueError("vector must contain finite values")
            self.db.execute("""INSERT INTO items(namespace,branch,item_key,text,vector,metadata,deleted)
                VALUES(?,?,?,?,?,?,0) ON CONFLICT(namespace,branch,item_key) DO UPDATE SET
                text=excluded.text, vector=excluded.vector, metadata=excluded.metadata, deleted=0""",
                (namespace, branch, key, new_text,
                 _json(new_vector) if new_vector is not None else None, _json(merged)))
            self.db.commit()
            now = datetime.now(timezone.utc).isoformat(); self.db.execute("UPDATE namespace_metadata SET last_write_at=?,updated_at=?,revision=revision+1 WHERE namespace=?", (now, now, namespace)); self.db.commit()
            self._invalidate_pins(namespace, branch)
            return current_revision + 1

    def update_metadata(self, namespace: str, key: str, metadata: Mapping[str, Any], *,
                        branch: str = "main", expected_revision: int | None = None) -> int:
        """Convenience alias for Vespa-style partial field updates."""
        return self.partial_update(namespace, key, branch=branch, metadata=metadata,
                                   expected_revision=expected_revision)

    def delete(self, namespace: str, key: str, *, branch: str = "main") -> None:
        with self._lock:
            self._ensure_branch(namespace, branch)
            meta_row = self.db.execute("SELECT read_only FROM namespace_metadata WHERE namespace=?", (namespace,)).fetchone()
            if meta_row and meta_row[0]: raise PermissionError(f"namespace is read-only: {namespace}")
            self.db.execute("""INSERT INTO items(namespace,branch,item_key,text,vector,metadata,deleted)
                VALUES(?,?,?, '', NULL, '{}', 1)
                ON CONFLICT(namespace,branch,item_key) DO UPDATE SET deleted=1""", (namespace, branch, key))
            self.db.commit()
            now = datetime.now(timezone.utc).isoformat(); self.db.execute("UPDATE namespace_metadata SET last_write_at=?,updated_at=?,revision=revision+1 WHERE namespace=?", (now, now, namespace)); self.db.commit()
            self._invalidate_pins(namespace, branch)

    def pin(self, namespace: str, branch: str = "main") -> int:
        with self._lock:
            snapshot = self._rows(namespace, branch)
            self._pinned[(namespace, branch)] = snapshot
            return len(snapshot)

    def prewarm(self, namespace: str, branch: str = "main") -> dict[str, Any]:
        """Warm a namespace snapshot before a latency-sensitive burst."""
        started = datetime.now(timezone.utc)
        rows = self.pin(namespace, branch)
        elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
        return {"namespace": namespace, "branch": branch, "rows_warmed": rows,
                "prewarm_ms": elapsed_ms, "cache_temperature": "hot"}

    def unpin(self, namespace: str, branch: str = "main") -> None:
        with self._lock:
            self._pinned.pop((namespace, branch), None)

    def _snapshot(self, namespace: str, branch: str) -> dict[str, dict[str, Any]]:
        with self._lock:
            key = (namespace, branch)
            if key in self._pinned:
                return self._pinned[key]
        return self._rows(namespace, branch)

    @staticmethod
    def _match_filter(metadata: Mapping[str, Any], filters: Any) -> bool:
        """Evaluate mapping filters and turbopuffer-style nested expressions."""
        if not filters:
            return True
        if isinstance(filters, Mapping):
            for key, expected in filters.items():
                actual = metadata.get(key)
                if isinstance(expected, Mapping):
                    if "in" in expected and actual not in expected["in"]: return False
                    if "contains" in expected:
                        values = actual if isinstance(actual, list) else [actual]
                        if not all(x in values for x in expected["contains"]): return False
                    if "eq" in expected and actual != expected["eq"]: return False
                elif isinstance(actual, list):
                    if isinstance(expected, list):
                        if not set(expected).issubset(actual): return False
                    elif expected not in actual: return False
                elif actual != expected: return False
            return True
        if not isinstance(filters, (list, tuple)):
            return False
        if isinstance(filters[0], str) and filters[0].casefold() in {"and", "or"}:
            values = [LocalSearchBackend._match_filter(metadata, x) for x in filters[1]]
            return all(values) if filters[0].casefold() == "and" else any(values)
        if isinstance(filters[0], str) and filters[0].casefold() == "not":
            return not LocalSearchBackend._match_filter(metadata, filters[1])
        if len(filters) < 3:
            return False
        key, op, expected = str(filters[0]), str(filters[1]).casefold(), filters[2]
        actual = metadata.get(key)
        if op in {"eq", "equals"}: return actual == expected
        if op == "noteq": return actual != expected
        if op in {"in", "notin"}:
            result = actual in expected
            return not result if op == "notin" else result
        values = actual if isinstance(actual, list) else [actual]
        if op in {"contains", "containsany"}: return expected in values if op == "contains" else any(x in values for x in expected)
        if op == "notcontains": return expected not in values
        if op == "notcontainsany": return not any(x in values for x in expected)
        try:
            if op in {"lt", "lte", "gt", "gte"}:
                return {"lt": actual < expected, "lte": actual <= expected,
                        "gt": actual > expected, "gte": actual >= expected}[op]
        except TypeError:
            return False
        return False

    @staticmethod
    def _authorized(metadata: Mapping[str, Any], principal: str | None) -> bool:
        status = metadata.get("status")
        if status is not None and status != "active":
            return False
        if metadata.get("revoked") is True:
            return False
        acl = metadata.get("acl", ["*"])
        if isinstance(acl, str):
            acl = [acl]
        if principal is not None and "*" not in acl and principal not in acl:
            return False
        gate = metadata.get("surprise_gate")
        if gate:
            novelty = metadata.get("novelty_score")
            contradiction = metadata.get("contradiction_score")
            if novelty is None or contradiction is None:
                return False
            if novelty < gate.get("novelty_threshold", 0.0):
                return False
            if contradiction > gate.get("contradiction_threshold", 1.0):
                return False
        return True

    def search(self, namespace: str = "default", *, branch: str = "main", text: str = "",
               vector: Iterable[float] | None = None, regex: str | None = None,
               filters: Any = None, principal: str | None = None,
               top_k: int = 10, vector_weight: float = 0.65, lexical_weight: float = 0.35,
               rank_profile: TensorRankProfile | None = None) -> list[SearchHit]:
        """Filter first, then fuse cosine, BM25-style lexical and regex scores."""
        if top_k <= 0: return []
        query_tokens = _tokens(text)
        snapshot = self._snapshot(namespace, branch)
        meta = self.namespace_metadata(namespace)
        fkey = (namespace, branch, str(meta.get("revision", 0)), _json(filters or {}), str(principal))
        allowed_keys = self._filter_cache.get(fkey)
        if allowed_keys is None:
            self.filter_cache_misses += 1
            allowed_keys = frozenset(x["key"] for x in snapshot.values() if not x["deleted"] and self._authorized(x["metadata"], principal)
                                     and self._match_filter(x["metadata"], filters or {}))
            self._filter_cache[fkey] = allowed_keys
        else:
            self.filter_cache_hits += 1
        candidates = [x for x in snapshot.values() if x["key"] in allowed_keys]
        # Reusable token postings narrow BM25 work before scoring; the index is
        # rebuilt only after writes/branch invalidation and is read-only on hot paths.
        pkey = (namespace, branch)
        postings = self._postings_cache.get(pkey)
        if postings is None:
            built = {}
            for item in snapshot.values():
                if item["deleted"]: continue
                for token in set(_tokens(item["text"])): built.setdefault(token, set()).add(item["key"])
            postings = {token: frozenset(keys) for token, keys in built.items()}; self._postings_cache[pkey] = postings
        if query_tokens:
            token_keys = set().union(*(postings.get(token, frozenset()) for token in query_tokens))
            # Keep the full allowed set while calculating IDF (otherwise a
            # selective query artificially gets document frequency one). Pure
            # lexical queries drop zero-score rows after scoring below.
        if regex:
            pattern = re.compile(regex, re.IGNORECASE)
            candidates = [x for x in candidates if pattern.search(x["text"])]
        else:
            pattern = None
        docs = [_tokens(x["text"]) for x in candidates]
        df = Counter(token for doc in docs for token in set(doc))
        n = len(candidates)
        scored: list[SearchHit] = []
        for item, doc in zip(candidates, docs):
            tf = Counter(doc)
            lexical = 0.0
            for token in query_tokens:
                if token in tf:
                    idf = math.log(1.0 + (n - df[token] + 0.5) / (df[token] + 0.5))
                    lexical += idf * (tf[token] * 2.2 / (tf[token] + 1.2))
            lexical = lexical / max(1.0, len(query_tokens))
            vector_score = max(0.0, _cosine(vector, item["vector"]))
            regex_score = 1.0 if pattern and pattern.search(item["text"]) else 0.0
            total = vector_weight * vector_score + lexical_weight * lexical + 0.15 * regex_score
            scored.append(SearchHit(item["key"], total, vector_score, lexical, regex_score, item["text"], dict(item["metadata"])))
        if vector is None and query_tokens and not regex:
            scored = [hit for hit in scored if hit.lexical_score > 0.0]
        if rank_profile is not None:
            scored = [SearchHit(hit.key, rank_profile.score(hit), hit.vector_score,
                                hit.lexical_score, hit.regex_score, hit.text, hit.metadata)
                      for hit in scored]
        scored.sort(key=lambda hit: (-hit.score, hit.key))
        return scored[:top_k]

    def query(self, namespace: str = "default", request: Mapping[str, Any] | None = None,
              *, branch: str = "main", principal: str | None = None) -> dict[str, Any]:
        """Execute the useful subset of turbopuffer's query contract locally.

        Supports ANN/BM25/attribute ordering, boolean filters, up to 16
        simultaneous subqueries and reciprocal-rank fusion.  Lifecycle and
        ACL checks still happen inside :meth:`search`, so a query expression
        cannot re-authorize revoked or stale Pods.
        """
        req = dict(request or {})

        def one(sub: Mapping[str, Any]) -> list[SearchHit]:
            rank = sub.get("rank_by", ["text", "BM25", ""])
            if isinstance(rank, list) and rank and str(rank[0]).casefold() == "sum":
                combined: dict[str, SearchHit] = {}; scores: dict[str, float] = {}
                for clause in rank[1] if len(rank) > 1 else []:
                    weight = 1.0
                    if isinstance(clause, list) and clause and str(clause[0]).casefold() == "product":
                        weight = float(clause[1]); clause = clause[2]
                    if not isinstance(clause, (list, tuple)): continue
                    child = dict(sub); child["rank_by"] = list(clause)
                    for hit in one(child):
                        combined.setdefault(hit.key, hit); scores[hit.key] = scores.get(hit.key, 0.0) + weight * hit.score
                return [SearchHit(h.key, scores[h.key], h.vector_score, h.lexical_score, h.regex_score, h.text, h.metadata)
                        for h in sorted(combined.values(), key=lambda h: (-scores[h.key], h.key))[:int(sub.get("top_k", sub.get("limit", 10)) if not isinstance(sub.get("limit", 10), Mapping) else sub["limit"].get("total", 10))]]
            text, vector, weights = "", None, (0.0, 1.0)
            if isinstance(rank, list) and len(rank) >= 3:
                field, fn, value = rank[:3]
                if str(fn).casefold() in {"ann", "knn"}:
                    if isinstance(value, (list, tuple)) and value and value[0] == "Embed":
                        if self.embedder is None: raise RuntimeError("native embedding requested but no local embedder is configured")
                        options = value[2] if len(value) > 2 and isinstance(value[2], Mapping) else {}
                        configured_model = self._embedding_config(namespace).get("model")
                        if options.get("model") and configured_model and options["model"] != configured_model:
                            raise ValueError(f"embedding model mismatch: configured {configured_model}, requested {options['model']}")
                        vector = self.embedder(str(value[1]))
                    else:
                        vector = value if isinstance(value, (list, tuple)) else None
                    weights = (1.0, 0.0)
                elif str(fn).casefold() == "bm25":
                    text, weights = str(value), (0.0, 1.0)
                elif str(fn).casefold() in {"asc", "desc"}:
                    text, weights = "", (0.0, 0.0)
            lim = sub.get("top_k", sub.get("limit", 10))
            lim = lim.get("total", 10) if isinstance(lim, Mapping) else lim
            hits = self.search(namespace, branch=branch, text=text, vector=vector,
                               filters=sub.get("filters"), principal=principal,
                               top_k=int(lim),
                               vector_weight=weights[0], lexical_weight=weights[1])
            if isinstance(rank, list) and len(rank) >= 2 and str(rank[1]).casefold() in {"asc", "desc"}:
                reverse = str(rank[1]).casefold() == "desc"
                hits.sort(key=lambda h: h.metadata.get(rank[0], h.key), reverse=reverse)
            return hits

        if "queries" in req:
            subs = list(req["queries"])
            if len(subs) > 16: raise ValueError("at most 16 subqueries are supported")
            # Independent subqueries are fanned out like a multi-query search
            # service.  SQLite serializes short snapshot reads safely while
            # scoring work can overlap; result order remains deterministic.
            workers = min(16, max(1, len(subs)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                lists = list(pool.map(one, subs))
            scores: dict[str, float] = {}
            by_key: dict[str, SearchHit] = {}
            rerank = req.get("rerank_by", ["RRF"])
            cfg = rerank[1] if isinstance(rerank, (list, tuple)) and len(rerank) > 1 and isinstance(rerank[1], Mapping) else {}
            constant = float(cfg.get("rank_constant", 60))
            weights = list(cfg.get("weights", []))
            for qi, rows in enumerate(lists):
                weight = float(weights[qi]) if qi < len(weights) else 1.0
                for rank, hit in enumerate(rows, 1):
                    by_key[hit.key] = hit
                    scores[hit.key] = scores.get(hit.key, 0.0) + weight / (constant + rank)
            rows = sorted(by_key.values(), key=lambda h: (-scores[h.key], h.key))
            lim = req.get("top_k", req.get("limit", 10)); lim = lim.get("total", 10) if isinstance(lim, Mapping) else lim
            return {"rows": [self._query_row(h, req, score=scores[h.key]) for h in rows[:int(lim)]], "rerank_by": req.get("rerank_by", ["RRF"])}
        if "aggregate_by" in req:
            visible = [x for x in self._snapshot(namespace, branch).values()
                       if not x["deleted"] and self._authorized(x["metadata"], principal)
                       and self._match_filter(x["metadata"], req.get("filters"))]
            aggregate = {}
            for label, expression in dict(req["aggregate_by"]).items():
                op = str(expression[0]).casefold() if expression else "count"
                aggregate[label] = sum(float(x["metadata"].get(expression[1], 0) or 0) for x in visible) if op == "sum" else len(visible)
            if req.get("group_by"):
                groups = {}
                for item in visible:
                    key = tuple(tuple(item["metadata"].get(k, [])) if isinstance(item["metadata"].get(k), list) else item["metadata"].get(k) for k in req["group_by"])
                    groups.setdefault(key, []).append(item)
                group_rows = []
                for key, members in sorted(groups.items(), key=lambda pair: repr(pair[0])):
                    row = {field: value for field, value in zip(req["group_by"], key)}
                    for label, expression in dict(req["aggregate_by"]).items():
                        row[label] = sum(float(x["metadata"].get(expression[1], 0) or 0) for x in members) if str(expression[0]).casefold() == "sum" else len(members)
                    group_rows.append(row)
                return {"aggregation_groups": group_rows[:int(req.get("limit", 10000))]}
            return {"aggregations": aggregate}
        rows = one(req)
        return {"rows": [self._query_row(h, req) for h in rows]}

    @staticmethod
    def _query_row(hit: SearchHit, request: Mapping[str, Any], *, score: float | None = None) -> dict[str, Any]:
        row = {"id": hit.key, "$dist": hit.score if score is None else score}
        include = request.get("include_attributes", True)
        exclude = set(request.get("exclude_attributes", []))
        if include is True:
            row.update(hit.metadata)
        elif include:
            row.update({key: hit.metadata[key] for key in include if key in hit.metadata})
        for key in exclude:
            row.pop(key, None)
        return row

    def stats(self, namespace: str, branch: str = "main") -> dict[str, Any]:
        snapshot = self._snapshot(namespace, branch)
        return {"namespace": namespace, "branch": branch, "items": sum(not x["deleted"] for x in snapshot.values()),
                "pinned": (namespace, branch) in self._pinned,
                "namespace_metadata": self.namespace_metadata(namespace)}
