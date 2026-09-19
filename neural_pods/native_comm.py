"""Native communication executor: a pod's small model speaks protocols.

The model is trained (LoRA) so its output text IS the protocol frame —
no tool-use schema, no JSON indirection. This module parses that output
fail-closed, enforces the pod's egress ACL (from its link contract), and
executes the frames natively over MQTT (via MeshEndpoint) or TCP sockets.

Dialect (one frame per line):
  PUB <topic> <json-payload>      MQTT publish
  SUB <topic-filter>              MQTT subscribe
  DIAL <host>:<port>              open a TCP connection
  SEND <base64-payload>           send on the dialed connection
  RECV                            receive and include in the result
  CLOSE                           close the dialed connection

Anything else is an invalid frame: counted, refused, never executed.
"""
from __future__ import annotations
import base64
import json
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_FRAMES = 16
MAX_PAYLOAD_BYTES = 64 * 1024


@dataclass(frozen=True)
class Frame:
    kind: str                      # PUB | SUB | DIAL | SEND | RECV | CLOSE
    args: tuple[str, ...]
    raw: str

    @property
    def valid(self) -> bool:
        return self.kind != "INVALID"


def parse_frames(text: str) -> list[Frame]:
    """Deterministic fail-closed parser: every non-empty line must be a
    valid frame, otherwise it becomes kind=INVALID (refused by the executor)."""
    frames = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        kind = parts[0].upper()
        if kind == "PUB" and len(parts) == 3 and _is_topic(parts[1]) and _is_json(parts[2]):
            frames.append(Frame("PUB", (parts[1], parts[2]), line))
        elif kind == "SUB" and len(parts) == 2 and _is_topic(parts[1]):
            frames.append(Frame("SUB", (parts[1],), line))
        elif kind == "DIAL" and len(parts) == 2 and re.fullmatch(r"[A-Za-z0-9.\-]+:\d{1,5}", parts[1]):
            frames.append(Frame("DIAL", (parts[1],), line))
        elif kind == "SEND" and len(parts) == 2 and _is_b64(parts[1]):
            frames.append(Frame("SEND", (parts[1],), line))
        elif kind == "RECV" and len(parts) == 1:
            frames.append(Frame("RECV", (), line))
        elif kind == "CLOSE" and len(parts) == 1:
            frames.append(Frame("CLOSE", (), line))
        else:
            frames.append(Frame("INVALID", (line,), line))
    return frames[:MAX_FRAMES]


def _is_topic(topic: str) -> bool:
    return bool(re.fullmatch(r"np/[A-Za-z0-9_\-+#/]{1,120}", topic))


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _is_b64(text: str) -> bool:
    try:
        return len(base64.b64decode(text, validate=True)) <= MAX_PAYLOAD_BYTES
    except Exception:
        return False


class EgressACL:
    """Pod egress allowlist from the link contract: which MQTT topics and
    host:port pairs this pod may address. Anything else is refused."""

    def __init__(self, *, topics: tuple[str, ...] = (), hosts: tuple[str, ...] = (),
                 ports: tuple[int, ...] = (), principal: str = "local"):
        self.topics = tuple(topics)
        self.hosts = tuple(hosts)
        self.ports = tuple(ports)
        self.principal = principal
        self.violations = 0

    def check_topic(self, topic: str) -> bool:
        ok = any(topic == t or topic.startswith(t.rstrip("#") ) and
                 (t.endswith("#") or topic == t) for t in self.topics)
        if not ok:
            self.violations += 1
        return ok

    def check_host_port(self, host: str, port: int) -> bool:
        ok = host in self.hosts and port in self.ports
        if not ok:
            self.violations += 1
        return ok


@dataclass
class CommResult:
    frames: list[Frame] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    refused: int = 0
    invalid: int = 0
    recv_data: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"executed": self.executed, "refused": self.refused,
                "invalid": self.invalid, "recv_data": self.recv_data,
                "frame_count": len(self.frames)}


class NativeCommExecutor:
    """Parses the pod model's output, enforces the egress ACL, and executes
    frames natively: MQTT through a MeshEndpoint, TCP through real sockets."""

    def __init__(self, *, mesh: Any = None, acl: EgressACL,
                 connect_timeout_s: float = 5.0, rate_limit_frames_s: float = 50.0):
        self.mesh = mesh
        self.acl = acl
        self.connect_timeout_s = float(connect_timeout_s)
        self.rate_limit_frames_s = float(rate_limit_frames_s)
        self._sock: socket.socket | None = None
        self._last_send = 0.0
        self._rate_lock = threading.Lock()

    def execute(self, model_output: str) -> CommResult:
        result = CommResult()
        dial_host = dial_port = None
        for frame in parse_frames(model_output):
            result.frames.append(frame)
            if not frame.valid:
                result.invalid += 1
                continue
            if not self._rate_ok():
                result.refused += 1
                continue
            if frame.kind == "PUB":
                topic, payload = frame.args
                if not self.acl.check_topic(topic) or self.mesh is None:
                    result.refused += 1
                    continue
                self.mesh.publish_raw(topic, json.loads(payload))
                result.executed.append({"frame": "PUB", "topic": topic})
            elif frame.kind == "SUB":
                topic = frame.args[0]
                if not self.acl.check_topic(topic) or self.mesh is None:
                    result.refused += 1
                    continue
                self.mesh.subscribe(topic, lambda t, e: None)
                result.executed.append({"frame": "SUB", "topic": topic})
            elif frame.kind == "DIAL":
                host, port = frame.args[0].rsplit(":", 1)
                if not self.acl.check_host_port(host, int(port)):
                    result.refused += 1
                    continue
                try:
                    self._sock = socket.create_connection((host, int(port)),
                                                          timeout=self.connect_timeout_s)
                    dial_host, dial_port = host, int(port)
                    result.executed.append({"frame": "DIAL", "host": host, "port": int(port)})
                except OSError as error:
                    result.executed.append({"frame": "DIAL", "error": repr(error)})
            elif frame.kind == "SEND":
                if self._sock is None:
                    result.refused += 1
                    continue
                self._sock.sendall(base64.b64decode(frame.args[0]))
                result.executed.append({"frame": "SEND", "bytes": len(frame.args[0])})
            elif frame.kind == "RECV":
                if self._sock is None:
                    result.refused += 1
                    continue
                self._sock.settimeout(self.connect_timeout_s)
                data = self._sock.recv(MAX_PAYLOAD_BYTES)
                result.recv_data.append(data.decode("utf-8", errors="replace"))
                result.executed.append({"frame": "RECV", "bytes": len(data)})
            elif frame.kind == "CLOSE":
                if self._sock is not None:
                    self._sock.close()
                    self._sock = None
                result.executed.append({"frame": "CLOSE"})
        return result

    def _rate_ok(self) -> bool:
        # Throttle (short sleep) rather than refuse: a legitimate frame
        # sequence like DIAL/SEND/RECV/CLOSE executes within microseconds.
        with self._rate_lock:
            now = time.monotonic()
            wait = 1.0 / self.rate_limit_frames_s - (now - self._last_send)
            if wait > 0:
                time.sleep(min(wait, 0.5))
            self._last_send = time.monotonic()
            return True

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
