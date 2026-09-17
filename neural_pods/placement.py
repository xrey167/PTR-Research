"""Placement and fencing primitives inspired by TiKV/PD and Raft.

This is intentionally a control-plane component, not a replacement for a
complete Raft implementation. It keeps namespace/region placement separate
from storage, prefers failure-domain diversity, and exposes a fenced leader
lease so stale Pod writers cannot commit after leadership changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass
class ReplicaNode:
    node_id: str
    zone: str
    capacity: int = 1
    load: float = 0.0
    last_heartbeat: float = 0.0
    healthy: bool = True

    def live(self, now: float, timeout_s: float) -> bool:
        return self.healthy and (now - self.last_heartbeat) <= timeout_s


@dataclass(frozen=True)
class RegionPlacement:
    namespace: str
    region_id: str
    replicas: tuple[str, ...]
    leader: str
    epoch: int


class PlacementDriver:
    """Small deterministic placement driver for namespaces/Pod regions."""

    def __init__(self, *, heartbeat_timeout_s: float = 5.0):
        if heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be positive")
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._nodes: dict[str, ReplicaNode] = {}
        self._regions: dict[tuple[str, str], RegionPlacement] = {}
        self._epoch = 0
        self._lock = threading.RLock()

    def register(self, node_id: str, *, zone: str, capacity: int = 1) -> None:
        if not node_id or not zone or capacity < 1:
            raise ValueError("node_id, zone and positive capacity are required")
        with self._lock:
            now = time.monotonic()
            self._nodes[node_id] = ReplicaNode(node_id, zone, capacity,
                                               last_heartbeat=now)

    def heartbeat(self, node_id: str, *, load: float | None = None) -> None:
        with self._lock:
            node = self._nodes[node_id]
            node.last_heartbeat = time.monotonic()
            node.healthy = True
            if load is not None:
                if load < 0:
                    raise ValueError("load must be non-negative")
                node.load = float(load)

    def set_health(self, node_id: str, healthy: bool) -> None:
        with self._lock:
            self._nodes[node_id].healthy = bool(healthy)

    def assign(self, namespace: str, region_id: str, *, replication_factor: int = 3) -> RegionPlacement:
        if not namespace or not region_id or replication_factor < 1:
            raise ValueError("invalid region assignment")
        with self._lock:
            now = time.monotonic()
            candidates = [n for n in self._nodes.values()
                          if n.live(now, self.heartbeat_timeout_s)]
            if len(candidates) < replication_factor:
                raise RuntimeError("insufficient healthy replicas")
            # First choose one per zone, then fill remaining slots by load.
            candidates.sort(key=lambda n: (n.load, n.node_id))
            selected: list[ReplicaNode] = []
            zones: set[str] = set()
            for node in candidates:
                if node.zone not in zones:
                    selected.append(node); zones.add(node.zone)
                    if len(selected) == replication_factor:
                        break
            if len(selected) < replication_factor:
                for node in candidates:
                    if node not in selected:
                        selected.append(node)
                        if len(selected) == replication_factor:
                            break
            self._epoch += 1
            placement = RegionPlacement(namespace, region_id,
                                         tuple(n.node_id for n in selected),
                                         selected[0].node_id, self._epoch)
            self._regions[(namespace, region_id)] = placement
            return placement

    def rebalance(self, namespace: str, region_id: str) -> RegionPlacement:
        with self._lock:
            current = self._regions[(namespace, region_id)]
            return self.assign(namespace, region_id,
                               replication_factor=len(current.replicas))

    def reconcile(self) -> list[RegionPlacement]:
        """Run one PD-style health pass and move regions off failed replicas."""
        changed: list[RegionPlacement] = []
        with self._lock:
            now = time.monotonic()
            keys = list(self._regions)
            for namespace, region_id in keys:
                current = self._regions[(namespace, region_id)]
                if any(not self._nodes[node].live(now, self.heartbeat_timeout_s)
                       for node in current.replicas):
                    changed.append(self.rebalance(namespace, region_id))
        return changed

    def route(self, namespace: str, region_id: str, *, write: bool = False) -> str:
        with self._lock:
            placement = self._regions[(namespace, region_id)]
            now = time.monotonic()
            if write:
                leader = self._nodes[placement.leader]
                if not leader.live(now, self.heartbeat_timeout_s):
                    raise RuntimeError("region leader is not healthy")
                return placement.leader
            for node_id in placement.replicas:
                if self._nodes[node_id].live(now, self.heartbeat_timeout_s):
                    return node_id
            raise RuntimeError("no healthy region replica")

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return {"epoch": self._epoch,
                    "nodes": {k: {"zone": v.zone, "load": v.load,
                                   "healthy": v.live(time.monotonic(), self.heartbeat_timeout_s)}
                               for k, v in self._nodes.items()},
                    "regions": {f"{ns}/{rid}": p.__dict__ for (ns, rid), p in self._regions.items()}}


@dataclass(frozen=True)
class LeaderLease:
    holder: str
    term: int
    fence: int
    expires_at: float


class FencedLeader:
    """In-process leader lease with monotonic term/fencing tokens."""

    def __init__(self, *, lease_s: float = 5.0):
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        self.lease_s = lease_s
        self._lease: LeaderLease | None = None
        self._term = 0
        self._fence = 0
        self._lock = threading.Lock()

    def acquire(self, holder: str) -> LeaderLease:
        if not holder:
            raise ValueError("holder is required")
        with self._lock:
            now = time.monotonic()
            if self._lease is not None and self._lease.expires_at > now and self._lease.holder != holder:
                raise RuntimeError("leader lease is held")
            self._term += 1; self._fence += 1
            self._lease = LeaderLease(holder, self._term, self._fence, now + self.lease_s)
            return self._lease

    def renew(self, holder: str, fence: int) -> LeaderLease:
        with self._lock:
            current = self._lease
            if current is None or current.holder != holder or current.fence != fence:
                raise RuntimeError("invalid leader fence")
            self._lease = LeaderLease(holder, current.term, current.fence,
                                      time.monotonic() + self.lease_s)
            return self._lease

    def validate(self, holder: str, fence: int) -> bool:
        with self._lock:
            return bool(self._lease and self._lease.holder == holder
                        and self._lease.fence == fence
                        and self._lease.expires_at > time.monotonic())
