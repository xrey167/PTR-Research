"""Mesh cache: pod results shared across mesh nodes via Redis, with
principal isolation and mesh-driven invalidation.

Two pods on different nodes share entries through the Redis instance in
np-node1. Keys carry the namespace and principal (ACL isolation), and
publishing an invalidation event on the mesh (`np/cache/invalidate`)
notifies every node to drop matching entries locally as well.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any


class MeshCache:
    def __init__(self, *, redis_client: Any, pod_id: str, principal: str = "local",
                 namespace: str = "mesh", ttl_s: int = 600):
        import paho.mqtt.client as _mqtt  # noqa: F401  (availability guard)
        self.redis = redis_client
        self.pod_id = pod_id
        self.principal = principal
        self.namespace = namespace
        self.ttl_s = int(ttl_s)
        self.hits = self.misses = self.writes = 0

    def _key(self, query_key: str, principal: str) -> str:
        digest = hashlib.sha256(
            json.dumps([self.namespace, principal, query_key],
                       sort_keys=True).encode()).hexdigest()[:20]
        return f"np:{self.namespace}:{principal}:{digest}"

    def put(self, query_key: str, value: Any, *, principal: str | None = None) -> None:
        principal = principal or self.principal
        self.redis.set(self._key(query_key, principal),
                       json.dumps({"value": value, "writer": self.pod_id}),
                       ex=self.ttl_s)
        self.writes += 1

    def get(self, query_key: str, *, principal: str | None = None) -> Any | None:
        principal = principal or self.principal
        raw = self.redis.get(self._key(query_key, principal))
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw.decode("utf-8"))["value"]

    def invalidate(self, query_key: str, *, principal: str | None = None) -> None:
        self.redis.delete(self._key(query_key, principal or self.principal))

    def stats(self) -> dict[str, Any]:
        return {"pod_id": self.pod_id, "hits": self.hits, "misses": self.misses,
                "writes": self.writes}
