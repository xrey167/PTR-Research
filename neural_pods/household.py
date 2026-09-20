"""Household swarm: pods behave like a Colibri/Lumabri household.

Pods join a household (key-bound), offer RAM/VRAM budgets, approve
allocation requests, and receive segment plans — a model's segments
(adapter shards, layer ranges, rerankers) distributed across donors,
size-verified before start. Busy donors refuse new work without replacing
running requests. Calibration records short measured probes per
pod+model (lumabri: "8 tokens, 20 s limit" — an indication, not a
guarantee). Approvals are provenance events, not silent flags.

Training (H5) and token caching (H6) ride on the same contracts: a
training request is an allocation with a Dream-Pod-coupled autonomy
budget; the TokenCache shares reusable prompt-prefix tokens across pods
via the L1 Redis tier.
"""
from __future__ import annotations
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import registry as registry_mod


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


@dataclass
class Offer:
    pod_id: str
    ram_bytes: int
    vram_bytes: int
    busy: bool = False
    busy_with: str | None = None
    calibrations: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"pod_id": self.pod_id, "ram_bytes": self.ram_bytes,
                "vram_bytes": self.vram_bytes, "busy": self.busy,
                "busy_with": self.busy_with,
                "calibrations": self.calibrations}


@dataclass
class Segment:
    name: str
    bytes_needed: int
    tier_preference: tuple[str, ...] = ("vram", "ram")   # colibri hierarchy


@dataclass
class AllocationRequest:
    request_id: str
    model_ref: str
    segments: tuple[Segment, ...]
    donors: tuple[str, ...]
    approved: tuple[str, ...] = ()
    status: str = "pending"          # pending | approved | started | released | refused
    plan_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "model_ref": self.model_ref,
                "segments": [vars(s) for s in self.segments],
                "donors": list(self.donors), "approved": list(self.approved),
                "status": self.status, "plan_hash": self.plan_hash}


@dataclass
class TokenCacheEntry:
    session_id: str
    prefix_hash: str
    token_count: int
    replica: str


class Household:
    """Donor pod swarm with approval-gated allocation.

    Zero-cost rule: a pod restart loses nothing — household state is
    reconstructible from the registry events and re-announced offers. That
    holds because every step that changes who owes what is an event:
    request, approval, refusal, start AND release. `restore_from_events()`
    replays them; before release existed, a donor that had once been used
    stayed BUSY forever and nothing in the log said which donors a start had
    claimed, so the rule was a claim rather than a property.
    """

    def __init__(self, key: str, registry: registry_mod.Registry):
        self.key_hash = _hash({"household_key": key})
        self.registry = registry
        self.offers: dict[str, Offer] = {}
        self.reservations: dict[str, dict] = {}
        self.requests: dict[str, AllocationRequest] = {}
        self.token_cache: dict[tuple[str, str], TokenCacheEntry] = {}
        self.saved_tokens = 0
        self._lock = threading.RLock()

    # --- H1: join / offers / members ------------------------------------
    def join(self, pod_id: str, key: str, *, ram_bytes: int, vram_bytes: int) -> Offer:
        if _hash({"household_key": key}) != self.key_hash:
            raise PermissionError("household key mismatch")
        offer = Offer(pod_id=pod_id, ram_bytes=ram_bytes, vram_bytes=vram_bytes)
        with self._lock:
            self.offers[pod_id] = offer
        return offer

    def members(self) -> list[dict[str, Any]]:
        with self._lock:
            return [o.to_dict() for o in self.offers.values()]

    # --- H2: allocation with approvals and BUSY --------------------------
    def request_allocation(self, model_ref: str, segments: list[Segment],
                           donors: list[str]) -> AllocationRequest:
        with self._lock:
            for pod_id in donors:
                offer = self.offers.get(pod_id)
                if offer is None:
                    raise KeyError(f"unknown donor: {pod_id}")
                if offer.busy:
                    # A refusal is part of the contract (ADR-6: BUSY replaces
                    # the silent override), so it belongs in the event log
                    # like the approvals do.
                    with self.registry.transaction():
                        self.registry._event("allocation_refused",
                                             {"model_ref": model_ref,
                                              "pod": pod_id,
                                              "busy_with": offer.busy_with})
                    return AllocationRequest(
                        request_id="BUSY", model_ref=model_ref,
                        segments=tuple(segments), donors=tuple(donors),
                        status="refused")
            request_id = f"alloc-{_hash({'model': model_ref, 'time': time.time()})}"
            plan = {"model_ref": model_ref,
                    "segments": [vars(s) for s in segments], "donors": donors}
            request = AllocationRequest(
                request_id=request_id, model_ref=model_ref,
                segments=tuple(segments), donors=tuple(donors),
                plan_hash=_hash(plan))
            self.requests[request_id] = request
            with self.registry.transaction():
                self.registry._event("allocation_request",
                                     {"request": request_id, "plan_hash": request.plan_hash})
            return request

    def approve(self, request_id: str, pod_id: str, key: str) -> str:
        """Record one donor's consent.

        The household key is required: an approval commits a donor's memory,
        so knowing a request id and a pod id must not be enough to give it.
        Runs under the household lock — `approved` is rebuilt as a new tuple,
        so two concurrent approvals could otherwise drop one of them.
        """
        if _hash({"household_key": key}) != self.key_hash:
            raise PermissionError("household key mismatch")
        with self._lock:
            request = self.requests[request_id]
            if pod_id not in request.donors:
                raise PermissionError(f"{pod_id} is not a donor of {request_id}")
            if pod_id in request.approved:
                return request.status
            request.approved = (*request.approved, pod_id)
            if set(request.approved) == set(request.donors):
                request.status = "approved"
            status = request.status
            plan_hash = request.plan_hash
        with self.registry.transaction():
            self.registry._event("allocation_approved",
                                 {"request": request_id, "pod": pod_id,
                                  "plan_hash": plan_hash})
        return status

    def start(self, request_id: str) -> dict[str, Any]:
        """Start only when EVERY donor approved; verify segment sizes fit the
        offered budgets (tier preference vram -> ram); mark donors BUSY.

        VRAM and RAM are counted separately. They used to share one counter,
        so a segment placed in a donor's VRAM also consumed that donor's RAM
        budget and the two tiers of ADR-7 collapsed into one pool. The tier
        preference is also honoured strictly now: a segment that asks only
        for vram is no longer silently placed in ram.
        """
        request = self.requests[request_id]
        if request.status != "approved":
            raise PermissionError("allocation not fully approved")
        plan: list[dict[str, Any]] = []
        claimed: dict[str, dict[str, int]] = {}
        capacity = {"vram": lambda offer: offer.vram_bytes,
                    "ram": lambda offer: offer.ram_bytes}
        with self._lock:
            # Place ALL segments first (a donor's remaining budget shrinks as
            # segments of the SAME request claim it), then mark donors BUSY.
            for segment in request.segments:
                placed = False
                for pod_id in request.donors:
                    offer = self.offers[pod_id]
                    if offer.busy:
                        continue
                    used = claimed.setdefault(pod_id, {"vram": 0, "ram": 0})
                    for tier in segment.tier_preference:
                        if tier not in capacity:
                            raise ValueError(f"unknown tier {tier!r} on segment "
                                             f"{segment.name}")
                        if used[tier] + segment.bytes_needed <= capacity[tier](offer):
                            used[tier] += segment.bytes_needed
                            plan.append({"segment": segment.name, "pod": pod_id,
                                         "tier": tier, "bytes": segment.bytes_needed})
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    raise RuntimeError(f"no donor fits segment {segment.name}")
            for pod_id in {entry["pod"] for entry in plan}:
                self.offers[pod_id].busy = True
                self.offers[pod_id].busy_with = request_id
            self.reservations[request_id] = {"plan": plan,
                                             "started_at": time.time()}
            request.status = "started"
        with self.registry.transaction():
            # The start is the step that actually commits a donor's memory.
            # Without it in the log the request and its approvals were
            # auditable but the reservation they led to was not.
            self.registry._event("allocation_started",
                                 {"request": request_id, "plan": plan,
                                  "plan_hash": request.plan_hash})
        return {"request_id": request_id, "plan": plan}

    def release(self, request_id: str, key: str) -> dict[str, Any]:
        """Hand a started allocation back; its donors stop being BUSY.

        Nothing used to clear `busy`. A donor that had served one allocation
        refused every later one for the lifetime of the process, so the
        household could serve exactly one request per donor, ever.

        The household key is required for the same reason it is on approve():
        releasing someone else's allocation takes their reservation away.
        """
        if _hash({"household_key": key}) != self.key_hash:
            raise PermissionError("household key mismatch")
        with self._lock:
            request = self.requests.get(request_id)
            if request is None:
                raise KeyError(f"unknown request: {request_id}")
            if request.status != "started":
                raise PermissionError(
                    f"request {request_id} is {request.status}, not started")
            reservation = self.reservations.pop(request_id, {"plan": []})
            freed = sorted({entry["pod"] for entry in reservation["plan"]
                            if self.offers.get(entry["pod"]) is not None
                            and self.offers[entry["pod"]].busy_with == request_id})
            for pod_id in freed:
                self.offers[pod_id].busy = False
                self.offers[pod_id].busy_with = None
            request.status = "released"
        with self.registry.transaction():
            self.registry._event("allocation_released",
                                 {"request": request_id, "pods": freed})
        return {"request_id": request_id, "released": freed}

    def restore_from_events(self) -> dict[str, Any]:
        """Rebuild reservations and BUSY flags from the provenance log.

        Offers are re-announced by the pods themselves via join(). What no
        pod can re-announce is which request currently holds its memory —
        that lives only in the started/released events, which is why the
        zero-cost rule needed them to exist.
        """
        open_plans: dict[str, list[dict]] = {}
        for event in reversed(self.registry.events(limit=1_000_000)):
            payload = event["payload"]
            if event["action"] == "allocation_started":
                open_plans[payload["request"]] = payload.get("plan", [])
            elif event["action"] == "allocation_released":
                open_plans.pop(payload["request"], None)
        with self._lock:
            self.reservations = {request_id: {"plan": plan, "started_at": None}
                                 for request_id, plan in open_plans.items()}
            for offer in self.offers.values():
                offer.busy = False
                offer.busy_with = None
            for request_id, plan in open_plans.items():
                for pod_id in {entry["pod"] for entry in plan}:
                    offer = self.offers.get(pod_id)
                    if offer is not None:
                        offer.busy = True
                        offer.busy_with = request_id
            busy = sorted(pod_id for pod_id, offer in self.offers.items() if offer.busy)
        return {"restored_requests": sorted(open_plans), "busy_pods": busy}

    # --- H3: calibration ---------------------------------------------------
    def calibrate(self, pod_id: str, model_ref: str, tokens_per_s: float,
                  *, tokens: int = 8) -> dict:
        """Record a short measured probe. An indication, not a guarantee."""
        offer = self.offers[pod_id]
        calibration = {"model_ref": model_ref, "tokens": tokens,
                       "tokens_per_s": round(tokens_per_s, 2),
                       "measured_at": time.time()}
        offer.calibrations[f"{model_ref}@{pod_id}"] = calibration
        return calibration

    # --- H6: token cache -----------------------------------------------------
    def token_cache_lookup(self, session_id: str, prefix_hash: str) -> TokenCacheEntry | None:
        entry = self.token_cache.get((session_id, prefix_hash))
        if entry is not None:
            self.saved_tokens += entry.token_count
        return entry

    def token_cache_store(self, session_id: str, prefix_hash: str,
                          token_count: int, replica: str) -> TokenCacheEntry:
        entry = TokenCacheEntry(session_id=session_id, prefix_hash=prefix_hash,
                                token_count=token_count, replica=replica)
        self.token_cache[(session_id, prefix_hash)] = entry
        return entry

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"members": len(self.offers),
                    "busy_pods": sum(1 for o in self.offers.values() if o.busy),
                    "requests": len(self.requests),
                    "reservations": len(self.reservations),
                    "saved_tokens": self.saved_tokens,
                    "key_hash": self.key_hash}
