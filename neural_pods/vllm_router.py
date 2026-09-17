"""Small dependency-free router for local vLLM OpenAI-compatible replicas."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import ssl
import urllib.error
import urllib.request
from typing import Any


@dataclass(frozen=True)
class VllmReplica:
    name: str
    base_url: str

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    @property
    def health_url(self) -> str:
        return self.base_url.rstrip("/") + "/health"


@dataclass
class VllmRouterMetrics:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    failovers: int = 0
    per_replica: dict[str, dict[str, int]] = field(default_factory=dict)


class VllmReplicaRouter:
    """Round-robin router with bounded retry and per-replica accounting.

    The router is intentionally transport-only: generation policy, auth headers,
    and model-specific payloads remain caller controlled.
    """

    def __init__(self, replicas: list[VllmReplica], *, timeout_s: float = 60.0,
                 max_attempts: int | None = None,
                 ssl_context: ssl.SSLContext | None = None):
        if not replicas:
            raise ValueError("at least one vLLM replica is required")
        self.replicas = tuple(replicas)
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts or len(self.replicas)
        self.ssl_context = ssl_context
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._cursor = 0
        self._lock = threading.Lock()
        self.metrics = VllmRouterMetrics(
            per_replica={r.name: {"requests": 0, "successes": 0, "failures": 0}
                         for r in self.replicas})

    def _ordered(self) -> list[VllmReplica]:
        with self._lock:
            start = self._cursor
            self._cursor = (self._cursor + 1) % len(self.replicas)
        return [self.replicas[(start + i) % len(self.replicas)] for i in range(len(self.replicas))]

    def health(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for replica in self.replicas:
            try:
                with urllib.request.urlopen(replica.health_url, timeout=self.timeout_s,
                                            context=self.ssl_context) as response:
                    result[replica.name] = 200 <= response.status < 300
            except (OSError, urllib.error.URLError):
                result[replica.name] = False
        return result

    def chat(self, payload: dict[str, Any], *, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Send one JSON chat request, retrying each selected replica at most once."""
        body = json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        ordered = self._ordered()[:self.max_attempts]
        with self._lock:
            self.metrics.requests += 1
        last_error: Exception | None = None
        for attempt, replica in enumerate(ordered):
            with self._lock:
                self.metrics.per_replica[replica.name]["requests"] += 1
            try:
                request = urllib.request.Request(replica.chat_url, body, request_headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_s,
                                            context=self.ssl_context) as response:
                    result = json.loads(response.read().decode("utf-8"))
                with self._lock:
                    self.metrics.successes += 1
                    self.metrics.per_replica[replica.name]["successes"] += 1
                    if attempt:
                        self.metrics.failovers += 1
                return result
            except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                with self._lock:
                    self.metrics.per_replica[replica.name]["failures"] += 1
        with self._lock:
            self.metrics.failures += 1
        raise RuntimeError(f"all vLLM replicas failed after {len(ordered)} attempts") from last_error
