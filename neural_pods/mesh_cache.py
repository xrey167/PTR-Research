"""Mesh cache: pod results shared across mesh nodes via Redis, with
principal-scoped keys and optional mesh-announced invalidation.

Two pods on different nodes share entries through the Redis instance in
np-node1. Keys carry the namespace and principal.

WHAT THE PRINCIPAL SCOPE IS, AND WHAT IT IS NOT. It is client-side key
derivation plus a client-side check: an instance serves the principal it was
constructed for and any further principal named in `allowed_principals`, and
refuses every other one (counted as `principal_refusals`). That bounds what
a *cooperating* caller can reach through this class.

It is NOT a security boundary, and the docstring used to let "ACL isolation"
imply that it was. Every principal's entries live in one Redis database
behind one credential; any process holding that credential — including a
second MeshCache constructed for the other principal, which is exactly what
`research/benchmark_mesh_cache.py` did while recording `principal_isolated:
true` — reads them with a plain GET. Enforcing isolation across principals
requires Redis ACLs (one user per principal, key patterns per user) or one
database per principal. Until that exists, `stats()["isolation"]` says
`client-side key derivation`, and no evidence file may claim more.

On invalidation, precisely: the shared Redis entry is deleted, which is
what makes an invalidation visible to every node that reads through this
cache. If a `mesh` endpoint is supplied, the deletion is ALSO announced on
`np/cache/invalidate`, so nodes holding their own in-process copies can
drop them; `subscribe_invalidations()` is the receiving side.

The previous docstring described that announcement as a fact while the code
only deleted the key, and imported paho purely so the dependency would look
used. Without a mesh endpoint this class now says so in `stats()` rather
than implying a mechanism it does not have.

Counters are taken under a lock: `invalidations_received` is incremented on
the mesh network thread, and `+=` on an attribute is not atomic.
"""
from __future__ import annotations
import hashlib
import json
import threading
from typing import Any


class PrincipalRefused(PermissionError):
    """A principal this instance was not constructed to serve."""


class MeshCache:
    INVALIDATION_TOPIC = "np/cache/invalidate"

    def __init__(self, *, redis_client: Any, pod_id: str, principal: str = "local",
                 namespace: str = "mesh", ttl_s: int = 600, mesh: Any = None,
                 allowed_principals: tuple[str, ...] | None = None):
        self.redis = redis_client
        self.mesh = mesh
        self.invalidations_sent = 0
        self.invalidations_received = 0
        self.pod_id = pod_id
        self.principal = principal
        self.allowed_principals = frozenset(
            (principal,) + tuple(allowed_principals or ()))
        self.namespace = namespace
        self.ttl_s = int(ttl_s)
        self.hits = self.misses = self.writes = 0
        self.decode_errors = 0
        self.principal_refusals = 0
        self._lock = threading.Lock()

    def _principal(self, principal: str | None) -> str:
        """Resolve and authorise the principal for one operation."""
        principal = principal or self.principal
        if principal not in self.allowed_principals:
            with self._lock:
                self.principal_refusals += 1
            raise PrincipalRefused(
                f"pod {self.pod_id!r} may serve "
                f"{sorted(self.allowed_principals)}, not {principal!r}")
        return principal

    def _key(self, query_key: str, principal: str) -> str:
        digest = hashlib.sha256(
            json.dumps([self.namespace, principal, query_key],
                       sort_keys=True).encode()).hexdigest()[:20]
        return f"np:{self.namespace}:{principal}:{digest}"

    def put(self, query_key: str, value: Any, *, principal: str | None = None) -> None:
        principal = self._principal(principal)
        self.redis.set(self._key(query_key, principal),
                       json.dumps({"value": value, "writer": self.pod_id}),
                       ex=self.ttl_s)
        with self._lock:
            self.writes += 1

    def get(self, query_key: str, *, principal: str | None = None) -> Any | None:
        principal = self._principal(principal)
        raw = self.redis.get(self._key(query_key, principal))
        if raw is None:
            with self._lock:
                self.misses += 1
            return None
        # redis-py returns bytes by default and str with decode_responses=True;
        # the hard-coded .decode() broke against the second configuration.
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            value = json.loads(raw)["value"]
        except (ValueError, KeyError, TypeError):
            # Counted as a hit before it was decoded, the hit rate described
            # entries this cache could not actually serve.
            with self._lock:
                self.decode_errors += 1
            return None
        with self._lock:
            self.hits += 1
        return value

    def invalidate(self, query_key: str, *, principal: str | None = None) -> None:
        """Drop the shared entry and, with a mesh, announce it."""
        principal = self._principal(principal)
        self.redis.delete(self._key(query_key, principal))
        if self.mesh is not None:
            self.mesh.publish_raw(self.INVALIDATION_TOPIC, {
                "namespace": self.namespace, "principal": principal,
                "query_key": query_key, "by": self.pod_id})
            with self._lock:
                self.invalidations_sent += 1

    def subscribe_invalidations(self, on_drop: Any = None) -> None:
        """Receive other nodes' invalidations for this namespace.

        The shared entry is already gone when this arrives; `on_drop` is the
        hook for a node that also keeps an in-process copy.
        """
        if self.mesh is None:
            raise RuntimeError("no mesh endpoint: nothing to subscribe to")

        def _handler(_topic: str, envelope: dict[str, Any]) -> None:
            body = envelope.get("body") or {}
            if body.get("namespace") != self.namespace or body.get("by") == self.pod_id:
                return
            with self._lock:
                self.invalidations_received += 1
            if on_drop is not None:
                on_drop(body)

        self.mesh.subscribe(self.INVALIDATION_TOPIC, _handler)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"pod_id": self.pod_id, "hits": self.hits,
                    "misses": self.misses, "writes": self.writes,
                    "decode_errors": self.decode_errors,
                    "principal": self.principal,
                    "allowed_principals": sorted(self.allowed_principals),
                    "principal_refusals": self.principal_refusals,
                    # Stated, not implied: the principal scope is a key prefix
                    # plus a client-side check. Anything holding the Redis
                    # credential reads every principal's keys directly.
                    "isolation": "client-side key derivation",
                    # Stated, not implied: without a mesh, an invalidation is a
                    # Redis delete and nothing is announced to anyone.
                    "mesh_bound": self.mesh is not None,
                    "invalidations_sent": self.invalidations_sent,
                    "invalidations_received": self.invalidations_received}
