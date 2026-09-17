"""High-throughput, namespace/type/tag aware Pod result cache."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import hashlib, json
import threading, time
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class CacheKey:
    namespace: str
    branch: str
    pod_type: str | None
    tags: tuple[str, ...]
    query: str
    vector_digest: str
    filters: str
    principal: str | None
    generation: str | None
    rank_profile: str | None
    namespace_revision: str | None
    top_k: int


class PodCache:
    """Bounded LRU cache with O(1) hit lookup and explicit invalidation.

    Namespace, branch, Pod type and tags are part of the key. ``search`` adds
    hard metadata filters before calling the backend, so cached results cannot
    cross Pod classes or branches.
    """
    def __init__(self, max_entries: int = 4096, ttl_seconds: float = 60.0):
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("max_entries and ttl_seconds must be positive")
        self.max_entries, self.ttl = max_entries, float(ttl_seconds)
        self._data: OrderedDict[CacheKey, tuple[float, tuple[Any, ...]]] = OrderedDict()
        self._inflight: dict[CacheKey, threading.Event] = {}
        # Explicit hot entries.  Pinning is opt-in because a learned hot set
        # can overfit a prompt distribution; the default remains plain LRU.
        self._pinned: set[CacheKey] = set()
        self._lock = threading.RLock()
        self.hits = self.misses = 0
        self.evictions = 0

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def _key(self, namespace, branch, pod_type, tags, text, top_k, *, vector=None,
             filters=None, principal=None, generation=None, rank_profile=None,
             namespace_revision=None):
        raw = self._canonical(list(vector)) if vector is not None else ""
        digest = hashlib.sha256(raw.encode()).hexdigest() if raw else ""
        profile = getattr(rank_profile, "__dict__", rank_profile)
        return CacheKey(namespace, branch, pod_type, tuple(sorted(set(tags or ()))), text,
                        digest, self._canonical(filters or {}), principal, generation,
                        self._canonical(profile) if profile is not None else None,
                        str(namespace_revision) if namespace_revision is not None else None,
                        int(top_k))

    def search(self, backend, namespace="default", *, branch="main", text="", vector=None,
               pod_type=None, tags: Iterable[str] = (), filters=None, principal=None, top_k=10,
               generation=None, rank_profile=None, **kwargs):
        revision = None
        if hasattr(backend, "namespace_metadata"):
            try: revision = backend.namespace_metadata(namespace).get("revision")
            except (KeyError, AttributeError): pass
        self._drop_stale_revisions(namespace, revision)
        key = self._key(namespace, branch, pod_type, tags, text, top_k, vector=vector,
                        filters=filters, principal=principal, generation=generation,
                        rank_profile=rank_profile, namespace_revision=revision)
        while True:
            now = time.monotonic()
            with self._lock:
                hit = self._data.get(key)
                if hit and now - hit[0] < self.ttl:
                    self.hits += 1; self._data.move_to_end(key); return list(hit[1])
                owner = key not in self._inflight
                if owner:
                    self._inflight[key] = threading.Event(); self.misses += 1
                event = self._inflight[key]
            if owner: break
            event.wait(max(0.001, self.ttl))
        hard = dict(filters or {})
        if pod_type is not None: hard["type"] = pod_type
        if tags: hard["tags"] = {"contains": sorted(set(tags))}
        try:
            result = tuple(backend.search(namespace, branch=branch, text=text, vector=vector,
                                          filters=hard, principal=principal, top_k=top_k,
                                          rank_profile=rank_profile, **kwargs))
            with self._lock:
                self._data[key] = (time.monotonic(), result); self._data.move_to_end(key)
                while len(self._data) > self.max_entries:
                    # Keep explicitly validated hot entries resident.  If all
                    # entries are pinned we temporarily allow bounded growth;
                    # unpinning restores normal LRU eviction on the next write.
                    victim = next((candidate for candidate in self._data
                                   if candidate not in self._pinned), None)
                    if victim is None:
                        break
                    self._data.pop(victim, None)
                    self.evictions += 1
        finally:
            with self._lock:
                event = self._inflight.pop(key, None)
                if event: event.set()
        return list(result)

    def _drop_stale_revisions(self, namespace: str, revision: Any) -> None:
        """Remove old-generation entries before lookup, including pinned ones.

        Revision-bearing keys already prevent a stale result from being
        returned.  Removing old entries as soon as a reader observes the new
        revision also prevents explicit hot pinning from retaining obsolete
        generations indefinitely.
        """
        if revision is None:
            return
        with self._lock:
            for key in list(self._data):
                if key.namespace == namespace and key.namespace_revision != str(revision):
                    self._data.pop(key, None)
                    self._pinned.discard(key)

    def invalidate(self, *, namespace=None, branch=None, pod_type=None, tags: Iterable[str] = ()) -> int:
        tags = set(tags or ()); removed = 0
        with self._lock:
            for key in list(self._data):
                if namespace is not None and key.namespace != namespace: continue
                if branch is not None and key.branch != branch: continue
                if pod_type is not None and key.pod_type != pod_type: continue
                if tags and not tags.intersection(key.tags): continue
                del self._data[key]; self._pinned.discard(key); removed += 1
        return removed

    def pin_query(self, backend, namespace="default", *, branch="main", text="", vector=None,
                  pod_type=None, tags: Iterable[str] = (), filters=None, principal=None,
                  top_k=10, generation=None, rank_profile=None, **kwargs):
        """Warm and pin one exact query result as an explicit hot entry.

        This is intentionally separate from ``search`` so normal traffic
        cannot silently change routing/cache policy.  Callers should pin only
        after a held-out workload confirms the access pattern; namespace
        revisions still invalidate the entry through the normal key contract.
        """
        result = self.search(backend, namespace, branch=branch, text=text, vector=vector,
                             pod_type=pod_type, tags=tags, filters=filters,
                             principal=principal, top_k=top_k, generation=generation,
                             rank_profile=rank_profile, **kwargs)
        revision = None
        if hasattr(backend, "namespace_metadata"):
            try: revision = backend.namespace_metadata(namespace).get("revision")
            except (KeyError, AttributeError): pass
        key = self._key(namespace, branch, pod_type, tags, text, top_k, vector=vector,
                        filters=filters, principal=principal, generation=generation,
                        rank_profile=rank_profile, namespace_revision=revision)
        with self._lock:
            if key in self._data:
                self._pinned.add(key)
        return result

    def unpin(self, *, namespace=None, branch=None, pod_type=None, tags: Iterable[str] = ()) -> int:
        """Release matching hot entries; future inserts enforce the LRU bound."""
        tags = set(tags or ()); removed = 0
        with self._lock:
            for key in list(self._pinned):
                if namespace is not None and key.namespace != namespace: continue
                if branch is not None and key.branch != branch: continue
                if pod_type is not None and key.pod_type != pod_type: continue
                if tags and not tags.intersection(key.tags): continue
                self._pinned.remove(key); removed += 1
            while len(self._data) > self.max_entries:
                victim = next((candidate for candidate in self._data
                               if candidate not in self._pinned), None)
                if victim is None: break
                self._data.pop(victim, None); self.evictions += 1
        return removed

    def stats(self) -> Mapping[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {"entries": len(self._data), "hits": self.hits, "misses": self.misses,
                    "hit_rate": self.hits / total if total else 0.0,
                    "inflight": len(self._inflight), "max_entries": self.max_entries,
                    "pinned": len(self._pinned), "evictions": self.evictions}
