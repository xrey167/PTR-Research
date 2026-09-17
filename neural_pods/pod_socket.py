"""Small newline-framed Pod transport for local/remote tunnel benchmarks.

It deliberately reuses :mod:`pod_protocol`; replacing TCP with a TLS socket,
Unix socket, socat or gRPC does not change request validation or lineage.
"""
from __future__ import annotations
import json, socket, socketserver, threading
import ssl
from dataclasses import asdict
from typing import Any
from .pod_protocol import PodRequest, PodResponse, PodTransport


def _request_from_dict(data: dict[str, Any]) -> PodRequest:
    return PodRequest(**{k: data[k] for k in PodRequest.__dataclass_fields__ if k in data})


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        transport: PodTransport = self.server.transport  # type: ignore[attr-defined]
        for line in self.rfile:
            try:
                response = transport.dispatch(_request_from_dict(json.loads(line)))
                payload = asdict(response)
            except Exception as exc:
                payload = {"error": f"protocol_error:{type(exc).__name__}"}
            self.wfile.write((json.dumps(payload, separators=(",", ":"), default=str) + "\n").encode())
            self.wfile.flush()


class _TLSRequestServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    def get_request(self):
        sock, address = super().get_request()
        context = getattr(self, "ssl_context", None)
        return (context.wrap_socket(sock, server_side=True) if context else sock), address


class PodSocketServer:
    def __init__(self, transport: PodTransport, host: str = "127.0.0.1", port: int = 0,
                 *, ssl_context: ssl.SSLContext | None = None):
        self.transport, self.host = transport, host
        self.server = _TLSRequestServer((host, port), _Handler)
        self.server.daemon_threads = True; self.server.allow_reuse_address = True
        self.server.transport = transport  # type: ignore[attr-defined]
        self.server.ssl_context = ssl_context
        self.thread: threading.Thread | None = None

    @property
    def address(self): return self.server.server_address

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start(); return self.address

    def close(self):
        self.server.shutdown(); self.server.server_close()
        if self.thread: self.thread.join(timeout=2)


class PodSocketClient:
    def __init__(self, host: str, port: int, *, timeout: float = 5.0,
                 ssl_context: ssl.SSLContext | None = None):
        self.host, self.port, self.timeout, self.ssl_context = host, int(port), float(timeout), ssl_context

    def dispatch(self, request: PodRequest) -> PodResponse:
        request.validate()
        with PodSocketSession(self) as session: return session.dispatch(request)


class PodSocketSession:
    """Persistent connection for low-latency bursts and duplex-style turns."""
    def __init__(self, client: PodSocketClient): self.client, self.sock, self.reader = client, None, None

    def __enter__(self):
        raw = socket.create_connection((self.client.host, self.client.port), timeout=self.client.timeout)
        self.sock = self.client.ssl_context.wrap_socket(raw, server_hostname=self.client.host) if self.client.ssl_context else raw
        self.sock.settimeout(self.client.timeout); self.reader = self.sock.makefile("rb"); return self

    def dispatch(self, request: PodRequest) -> PodResponse:
        request.validate(); self.sock.sendall((json.dumps(asdict(request), separators=(",", ":"), default=str) + "\n").encode())
        data = json.loads(self.reader.readline())
        fields = {k: data.get(k) for k in PodResponse.__dataclass_fields__}
        fields["payload"] = fields.get("payload") or {}
        return PodResponse(**fields)

    def __exit__(self, exc_type, exc, tb):
        if self.reader: self.reader.close()
        if self.sock: self.sock.close()
