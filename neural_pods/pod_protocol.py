"""Small, transport-neutral protocol envelope for Pod-to-Pod calls.

The envelope is intentionally independent of HTTP, gRPC and vLLM.  A tunnel
or local call can use the same validation rules and preserve lineage, deadlines
and hop limits across every boundary.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Mapping
import uuid, hashlib, hmac, json
from concurrent.futures import ThreadPoolExecutor


@dataclass(frozen=True)
class PodRequest:
    source_pod: str
    target_pod: str
    capability: str
    payload: Mapping[str, Any]
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    deadline_ms: int = 5000
    hop_budget: int = 4
    visited: tuple[str, ...] = ()
    target_generation: str | None = None
    target_artifact: str | None = None
    principal: str = "local"
    epoch: int = 0
    manifest_hash: str | None = None
    signature: str | None = None

    def validate(self, *, now_ms: int | None = None) -> None:
        if not self.source_pod or not self.target_pod or not self.capability:
            raise ValueError("source_pod, target_pod and capability are required")
        if self.source_pod == self.target_pod or self.target_pod in self.visited:
            raise ValueError("Pod link cycle detected")
        if self.hop_budget < 1:
            raise ValueError("hop_budget exhausted")
        if self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if now_ms is not None and now_ms >= self.deadline_ms:
            raise TimeoutError("Pod request deadline exceeded")

    def signing_bytes(self) -> bytes:
        return json.dumps({"trace_id": self.trace_id, "request_id": self.request_id,
                           "source_pod": self.source_pod, "target_pod": self.target_pod,
                           "capability": self.capability, "payload": self.payload,
                           "target_generation": self.target_generation,
                           "target_artifact": self.target_artifact, "epoch": self.epoch},
                          sort_keys=True, separators=(",", ":"), default=str).encode()

    def sign(self, secret: bytes) -> "PodRequest":
        signature = hmac.new(secret, self.signing_bytes(), hashlib.sha256).hexdigest()
        return PodRequest(**{**self.__dict__, "signature": signature})

    def next_hop(self, target_pod: str) -> "PodRequest":
        self.validate()
        if target_pod in (*self.visited, self.source_pod, self.target_pod):
            raise ValueError("Pod link cycle detected")
        return PodRequest(source_pod=self.target_pod, target_pod=target_pod,
                          capability=self.capability, payload=self.payload,
                          trace_id=self.trace_id, request_id=self.request_id,
                          deadline_ms=self.deadline_ms, hop_budget=self.hop_budget - 1,
                          visited=(*self.visited, self.target_pod),
                          target_generation=self.target_generation,
                          target_artifact=self.target_artifact, principal=self.principal,
                          epoch=self.epoch, manifest_hash=self.manifest_hash,
                          signature=self.signature)


@dataclass(frozen=True)
class PodResponse:
    request_id: str
    trace_id: str
    source_pod: str
    target_pod: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    server_event_seq: int = 0
    epoch: int = 0
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class PodTransport:
    """In-process transport with deadline and capability validation.

    Production adapters can wrap this contract with Unix sockets, socat,
    gRPC or a vLLM endpoint without changing Pod payloads.
    """
    def __init__(self, *, secret: bytes | None = None, max_retries: int = 0,
                 tracker: Any | None = None):
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be between 0 and 5")
        self.secret, self.max_retries, self.tracker = secret, max_retries, tracker
        self._handlers: dict[tuple[str, str], Any] = {}
        self._manifests: dict[str, str | None] = {}

    def register(self, pod_id: str, capability: str, handler: Any, *, manifest_hash: str | None = None) -> None:
        self._handlers[(pod_id, capability)] = handler
        self._manifests[pod_id] = manifest_hash

    def dispatch(self, request: PodRequest) -> PodResponse:
        """Dispatch and optionally record a bounded lifecycle receipt."""
        if self.tracker is None:
            return self._dispatch(request)
        self.tracker.start(request.request_id, request.trace_id, request.target_pod,
                           request.target_generation)
        try:
            response = self._dispatch(request)
            self.tracker.phase(request.request_id, "dispatch", response.latency_ms)
            payload = response.payload if isinstance(response.payload, Mapping) else {}
            self.tracker.finish(
                request.request_id, success=response.ok, error=response.error,
                input_tokens=int(payload.get("input_tokens", 0) or 0),
                output_tokens=int(payload.get("output_tokens", 0) or 0),
                cache_hit=bool(payload.get("cache_hit", False)),
            )
            return response
        except BaseException as exc:
            # Keep tracker state bounded and close even unexpected handler errors.
            self.tracker.finish(request.request_id, success=False, error=type(exc).__name__)
            raise

    def _dispatch(self, request: PodRequest) -> PodResponse:
        started = monotonic()
        request.validate(now_ms=0)
        expected_manifest = self._manifests.get(request.target_pod)
        if expected_manifest and request.manifest_hash != expected_manifest:
            return PodResponse(request.request_id, request.trace_id, request.source_pod,
                               request.target_pod, error="manifest_mismatch")
        if self.secret is not None:
            expected_signature = hmac.new(self.secret, request.signing_bytes(), hashlib.sha256).hexdigest()
            if not request.signature or not hmac.compare_digest(request.signature, expected_signature):
                return PodResponse(request.request_id, request.trace_id, request.source_pod,
                                   request.target_pod, error="invalid_signature")
        handler = self._handlers.get((request.target_pod, request.capability))
        if handler is None:
            return PodResponse(request.request_id, request.trace_id, request.source_pod,
                               request.target_pod, error="capability_unavailable")
        try:
            attempts = self.max_retries + 1
            payload = None
            for attempt in range(attempts):
                if (monotonic() - started) * 1000.0 >= request.deadline_ms:
                    break
                try:
                    payload = handler(request.payload, request)
                    break
                except (TimeoutError, ConnectionError):
                    if attempt + 1 == attempts: raise
            elapsed = (monotonic() - started) * 1000.0
            if elapsed > request.deadline_ms:
                return PodResponse(request.request_id, request.trace_id, request.source_pod,
                                   request.target_pod, error="deadline_exceeded",
                                   latency_ms=elapsed)
            return PodResponse(request.request_id, request.trace_id, request.source_pod,
                               request.target_pod, payload=payload or {}, latency_ms=elapsed,
                               epoch=request.epoch)
        except TimeoutError:
            elapsed = (monotonic() - started) * 1000.0
            return PodResponse(request.request_id, request.trace_id, request.source_pod,
                               request.target_pod, error="deadline_exceeded",
                               latency_ms=elapsed)
        except Exception as exc:
            return PodResponse(request.request_id, request.trace_id, request.source_pod,
                               request.target_pod, error=f"handler_error:{type(exc).__name__}")


class PodFanout:
    """Bounded parallel dispatch for independent Pod branches.

    Results stay in request order, while one failed branch becomes an explicit
    response instead of cancelling unrelated work.  The caller can merge only
    successful, verified payloads with ``pod_streams.merge_branches``.
    """
    def __init__(self, transport: PodTransport, *, max_workers: int = 8):
        if max_workers < 1 or max_workers > 64:
            raise ValueError("max_workers must be between 1 and 64")
        self.transport, self.max_workers = transport, max_workers

    def dispatch(self, requests: list[PodRequest]) -> list[PodResponse]:
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(requests))) as pool:
            return list(pool.map(self.transport.dispatch, requests))
