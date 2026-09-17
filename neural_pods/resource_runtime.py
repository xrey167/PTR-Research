"""Colibri-inspired request telemetry and hardware-aware resource admission.

The runtime keeps three concerns separate:

* :func:`probe_hardware` records a point-in-time, truthful view of CPU/RAM,
  GPUs and disk.  A missing measurement is ``None`` and is never treated as
  free capacity.
* :class:`ResourceGovernor` accounts leases against explicit safety reserves
  before a Pod is placed on VRAM, RAM or disk.
* :class:`RequestTracker` records a bounded request lifecycle with phase
  timings and resource snapshots.  It is intentionally transport agnostic so
  the same receipt can follow an in-process, socket or remote Pod request.

This mirrors the useful Colibri properties without coupling neural-pods to its
C process or text telemetry format.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping


GIB = 1024 ** 3


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_bytes: int | None
    free_bytes: int | None
    used_bytes: int | None = None


@dataclass(frozen=True)
class HardwareSnapshot:
    captured_at: float
    cpu_logical: int
    cpu_physical: int | None
    ram_total_bytes: int | None
    ram_available_bytes: int | None
    gpus: tuple[GPUInfo, ...] = ()
    disk_free_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "cpu_logical": self.cpu_logical,
            "cpu_physical": self.cpu_physical,
            "ram_total_bytes": self.ram_total_bytes,
            "ram_available_bytes": self.ram_available_bytes,
            "gpus": [g.__dict__ for g in self.gpus],
            "disk_free_bytes": self.disk_free_bytes,
        }


def _memory() -> tuple[int | None, int | None]:
    try:
        import psutil  # type: ignore
        m = psutil.virtual_memory()
        return int(m.total), int(m.available)
    except (ImportError, OSError):
        return None, None


def _gpus() -> tuple[GPUInfo, ...]:
    """Read NVIDIA capacity without importing CUDA or changing device state."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    rows: list[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            idx = int(parts[0]); total = int(parts[2]) * 1024 ** 2
            used = int(parts[3]) * 1024 ** 2; free = int(parts[4]) * 1024 ** 2
        except ValueError:
            continue
        rows.append(GPUInfo(idx, parts[1], total, free, used))
    return tuple(rows)


def probe_hardware(*, path: str | os.PathLike[str] = ".") -> HardwareSnapshot:
    total, available = _memory()
    try:
        logical = int(os.cpu_count() or 1)
    except (TypeError, ValueError):
        logical = 1
    physical: int | None = None
    try:
        import psutil  # type: ignore
        physical = psutil.cpu_count(logical=False)
    except (ImportError, OSError):
        pass
    try:
        free = int(shutil.disk_usage(path).free)
    except OSError:
        free = None
    return HardwareSnapshot(time.time(), logical, physical, total, available, _gpus(), free)


@dataclass(frozen=True)
class ResourceBudget:
    ram_bytes: int | None
    vram_bytes: Mapping[int, int]
    disk_bytes: int | None

    @classmethod
    def from_snapshot(cls, snapshot: HardwareSnapshot, *, ram_reserve: float = .15,
                      vram_reserve_bytes: int = 2 * GIB,
                      disk_reserve_bytes: int = 10 * GIB) -> "ResourceBudget":
        if not 0 <= ram_reserve < 1:
            raise ValueError("ram_reserve must be in [0, 1)")
        ram = (int(snapshot.ram_available_bytes * (1 - ram_reserve))
               if snapshot.ram_available_bytes is not None else None)
        vr: dict[int, int] = {}
        for gpu in snapshot.gpus:
            if gpu.free_bytes is not None:
                vr[gpu.index] = max(0, gpu.free_bytes - vram_reserve_bytes)
        disk = (max(0, snapshot.disk_free_bytes - disk_reserve_bytes)
                if snapshot.disk_free_bytes is not None else None)
        return cls(ram, vr, disk)


@dataclass(frozen=True)
class ResourceLease:
    tier: str
    amount_bytes: int
    device: int | None = None


@dataclass(frozen=True)
class ResidencyLease:
    pod_id: str
    generation: str
    lease: ResourceLease


class ResourceGovernor:
    """Thread-safe admission controller for model/Pod residency."""
    def __init__(self, budget: ResourceBudget):
        self.budget = budget
        self._used: Counter[tuple[str, int | None]] = Counter()
        self._residencies: dict[str, ResidencyLease] = {}
        self._lock = threading.Lock()

    def _limit(self, tier: str, device: int | None) -> int | None:
        if tier == "ram": return self.budget.ram_bytes
        if tier == "disk": return self.budget.disk_bytes
        if tier == "vram": return self.budget.vram_bytes.get(device)
        raise ValueError(f"unknown resource tier: {tier}")

    def try_acquire(self, tier: str, amount_bytes: int, *, device: int | None = None) -> ResourceLease | None:
        if amount_bytes < 0: raise ValueError("amount_bytes must be non-negative")
        with self._lock:
            limit = self._limit(tier, device)
            key = (tier, device)
            if limit is None or self._used[key] + amount_bytes > limit:
                return None
            self._used[key] += amount_bytes
            return ResourceLease(tier, amount_bytes, device)

    def release(self, lease: ResourceLease) -> None:
        with self._lock:
            key = (lease.tier, lease.device)
            self._used[key] -= lease.amount_bytes
            if self._used[key] < 0:
                raise RuntimeError("resource lease released more than acquired")

    def activate(self, pod_id: str, generation: str, amount_bytes: int, *,
                 preferred: str = "vram") -> ResidencyLease | None:
        """Reserve residency for one Pod generation.

        A Pod identity cannot silently switch generations while resident. The
        caller must deactivate the old generation first, which mirrors the
        lifecycle barrier used by the semantic registry.
        """
        if not pod_id or not generation:
            raise ValueError("pod_id and generation are required")
        with self._lock:
            current = self._residencies.get(pod_id)
            if current is not None:
                if current.generation != generation:
                    raise RuntimeError("Pod has a different generation resident")
                return current
            # choose() takes the same lock, so perform the selection inline.
            order = {"vram": ("vram", "ram", "disk"),
                     "ram": ("ram", "disk"), "disk": ("disk",)}.get(preferred)
            if order is None:
                raise ValueError(f"unknown preferred tier: {preferred}")
            resource = None
            for tier in order:
                devices = tuple(self.budget.vram_bytes) if tier == "vram" else (None,)
                for device in devices:
                    limit = self._limit(tier, device)
                    key = (tier, device)
                    if limit is not None and self._used[key] + amount_bytes <= limit:
                        self._used[key] += amount_bytes
                        resource = ResourceLease(tier, amount_bytes, device)
                        break
                if resource is not None:
                    break
            if resource is None:
                return None
            residency = ResidencyLease(pod_id, generation, resource)
            self._residencies[pod_id] = residency
            return residency

    def deactivate(self, pod_id: str, generation: str | None = None) -> bool:
        """Release a resident Pod; reject stale-generation eviction."""
        with self._lock:
            residency = self._residencies.get(pod_id)
            if residency is None:
                return False
            if generation is not None and residency.generation != generation:
                raise RuntimeError("stale generation cannot evict active Pod")
            self._used[(residency.lease.tier, residency.lease.device)] -= residency.lease.amount_bytes
            self._residencies.pop(pod_id, None)
            return True

    def choose(self, amount_bytes: int, *, preferred: str = "vram") -> ResourceLease | None:
        order = {"vram": ("vram", "ram", "disk"), "ram": ("ram", "disk"),
                 "disk": ("disk",)}.get(preferred)
        if order is None: raise ValueError(f"unknown preferred tier: {preferred}")
        for tier in order:
            devices = tuple(self.budget.vram_bytes) if tier == "vram" else (None,)
            for device in devices:
                lease = self.try_acquire(tier, amount_bytes, device=device)
                if lease is not None: return lease
        return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"used": {f"{t}:{d}": n for (t, d), n in self._used.items() if n},
                    "limits": {"ram": self.budget.ram_bytes, "vram": dict(self.budget.vram_bytes),
                               "disk": self.budget.disk_bytes},
                    "residencies": {pod: {"generation": r.generation,
                                            "tier": r.lease.tier,
                                            "device": r.lease.device,
                                            "amount_bytes": r.lease.amount_bytes}
                                     for pod, r in self._residencies.items()}}


class ResourceAdmissionError(RuntimeError):
    """Raised when a Pod cannot be admitted within the measured budget."""


class ResourceBoundHandler:
    """Bind a Pod handler to an explicit resource lease.

    The lease is acquired before the request enters a batch queue and released
    after the handler returns, including timeout/error paths. This makes the
    governor enforceable at the actual Pod activation boundary rather than
    being telemetry only.
    """

    def __init__(self, handler: Any, governor: ResourceGovernor, *,
                 amount_bytes: int, preferred: str = "vram") -> None:
        if amount_bytes < 0:
            raise ValueError("amount_bytes must be non-negative")
        self.handler = handler
        self.governor = governor
        self.amount_bytes = amount_bytes
        self.preferred = preferred
        self.admitted = 0
        self.rejected = 0
        self._lock = threading.Lock()

    def __call__(self, payload: Any, request: Any) -> Any:
        lease = self.governor.choose(self.amount_bytes, preferred=self.preferred)
        if lease is None:
            with self._lock:
                self.rejected += 1
            raise ResourceAdmissionError("Pod request exceeds available resource budget")
        with self._lock:
            self.admitted += 1
        try:
            return self.handler(payload, request)
        finally:
            self.governor.release(lease)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"admitted": self.admitted, "rejected": self.rejected}


@dataclass
class RequestReceipt:
    request_id: str
    trace_id: str
    pod_id: str
    generation: str | None
    started_at: float
    ended_at: float | None = None
    success: bool | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    phases_ms: dict[str, float] = field(default_factory=dict)
    resource_start: dict[str, Any] | None = None
    resource_end: dict[str, Any] | None = None

    @property
    def latency_ms(self) -> float | None:
        return None if self.ended_at is None else (self.ended_at - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__); d["latency_ms"] = self.latency_ms; return d


class RequestTracker:
    """Bounded lifecycle history and aggregate metrics for Pod requests."""
    def __init__(self, *, max_records: int = 4096):
        if max_records < 1: raise ValueError("max_records must be positive")
        self._records: deque[RequestReceipt] = deque(maxlen=max_records)
        self._active: dict[str, RequestReceipt] = {}
        self._lock = threading.Lock()

    def start(self, request_id: str, trace_id: str, pod_id: str, generation: str | None = None,
              *, hardware: HardwareSnapshot | None = None) -> RequestReceipt:
        with self._lock:
            if request_id in self._active: raise ValueError("request already active")
            receipt = RequestReceipt(request_id, trace_id, pod_id, generation, time.time(),
                                     resource_start=hardware.to_dict() if hardware else None)
            self._active[request_id] = receipt
            return receipt

    def phase(self, request_id: str, name: str, elapsed_ms: float) -> None:
        if elapsed_ms < 0: raise ValueError("elapsed_ms must be non-negative")
        with self._lock:
            self._active[request_id].phases_ms[name] = float(elapsed_ms)

    def finish(self, request_id: str, *, success: bool, error: str | None = None,
               input_tokens: int = 0, output_tokens: int = 0, cache_hit: bool = False,
               hardware: HardwareSnapshot | None = None) -> RequestReceipt:
        with self._lock:
            receipt = self._active.pop(request_id)
            receipt.ended_at = time.time(); receipt.success = bool(success)
            receipt.error = error; receipt.input_tokens = int(input_tokens)
            receipt.output_tokens = int(output_tokens); receipt.cache_hit = bool(cache_hit)
            receipt.resource_end = hardware.to_dict() if hardware else None
            self._records.append(receipt)
            return receipt

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values: return 0.0
        values.sort(); return values[min(len(values)-1, int((len(values)-1) * q))]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._records); active = len(self._active)
        lat = [r.latency_ms for r in rows if r.latency_ms is not None]
        return {"completed": len(rows), "active": active,
                "success": sum(r.success is True for r in rows),
                "errors": sum(r.success is False for r in rows),
                "cache_hits": sum(r.cache_hit for r in rows),
                "input_tokens": sum(r.input_tokens for r in rows),
                "output_tokens": sum(r.output_tokens for r in rows),
                "latency_p50_ms": self._percentile([float(x) for x in lat], .50),
                "latency_p95_ms": self._percentile([float(x) for x in lat], .95),
                "latency_max_ms": max(lat, default=0.0)}

    def export_jsonl(self, path: str | os.PathLike[str]) -> None:
        with self._lock: rows = [r.to_dict() for r in self._records]
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically so a crash cannot leave a partial receipt log.
        tmp = target.with_name(target.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
        try:
            tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            os.replace(tmp, target)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
