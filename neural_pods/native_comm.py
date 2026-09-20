"""Native communication executor: a pod's small model speaks protocols.

The model is trained (LoRA) so its output text IS the protocol frame —
no tool-use schema, no JSON indirection. This module parses that output
fail-closed, enforces the pod's egress ACL, and executes the frames natively
over MQTT (via MeshEndpoint) or TCP sockets.

Dialect (one frame per line):
  PUB <topic> <json-payload>      MQTT publish
  SUB <topic-filter>              MQTT subscribe
  DIAL <host>:<port>              open a TCP connection
  SEND <base64-payload>           send on the dialed connection
  RECV                            receive and include in the result
  CLOSE                           close the dialed connection

WHAT FAIL-CLOSED MEANS HERE, since the word was doing more work than the
code. An invalid line has always been counted and never executed. But the
VALID lines around it still ran, so an output the parser did not understand
as a whole could still publish: `sudo rm -rf /` followed by a legitimate PUB
published. That is per-frame refusal, not fail-closed.

`strict=True` (the default) refuses the ENTIRE output when any line fails to
parse or when the model emitted more frames than MAX_FRAMES. One
misunderstood line means the output as a whole was not understood, and a
model that half-speaks the dialect is exactly the case this is for.
`strict=False` restores the per-frame behaviour and exists for
research/benchmark_native_comm.py, which measures how OFTEN the model gets a
frame right and therefore has to see the partial ones.

THE EGRESS ACL COMES FROM THE LINK CONTRACT, which it did not before: the
docstring said so while every ACL was hand-built at the call site and
`PodLink` carried no egress fields at all. `EgressACL.from_link()` is the
derivation, and `PodLink.egress_topics/_hosts/_ports` is what it reads.

REFUSALS ARE RECORDED. A refused frame is a pod trying to address something
it may not, which is the one event in this module worth keeping: pass a
`registry` and every violation becomes an `egress_refused` provenance event.
Without one the counters still move, and `stats()` says which of the two is
in force rather than implying the log exists.
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


def frame_overflow(text: str) -> int:
    """How many frames beyond MAX_FRAMES the model emitted."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return max(len(lines) - MAX_FRAMES, 0)


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
    """Pod egress allowlist: which MQTT topics and host:port pairs this pod
    may address. Anything else is refused, counted, and — with a registry —
    written to the provenance log."""

    def __init__(self, *, topics: tuple[str, ...] = (), hosts: tuple[str, ...] = (),
                 ports: tuple[int, ...] = (), principal: str = "local",
                 registry: Any = None, pod_id: str = "unknown",
                 trace_id: str | None = None):
        self.topics = tuple(topics)
        self.hosts = tuple(hosts)
        self.ports = tuple(ports)
        self.principal = principal
        self.registry = registry
        self.pod_id = pod_id
        self.trace_id = trace_id
        self.violations = 0
        self.recorded_violations = 0

    @classmethod
    def from_link(cls, link: Any, *, registry: Any = None,
                  pod_id: str | None = None) -> "EgressACL":
        """Build the ACL a link contract authorises.

        This is the derivation the module docstring always claimed. A link
        with no egress fields yields an ACL that permits nothing, which is
        the correct reading of a contract that grants nothing — not a reason
        to fall back to an open one.
        """
        return cls(topics=tuple(getattr(link, "egress_topics", ()) or ()),
                   hosts=tuple(getattr(link, "egress_hosts", ()) or ()),
                   ports=tuple(getattr(link, "egress_ports", ()) or ()),
                   principal=(link.acl[0] if getattr(link, "acl", ()) else "local"),
                   registry=registry,
                   pod_id=pod_id or getattr(link, "source_pod_id", "unknown"),
                   trace_id=getattr(link, "trace_id", None))

    def _refuse(self, kind: str, target: str) -> None:
        self.violations += 1
        if self.registry is None:
            return
        # A refusal that leaves no trace cannot be reviewed afterwards, and a
        # pod repeatedly addressing what it may not is precisely the pattern
        # someone would want to find in the log later.
        self.registry.record_event("egress_refused", {
            "pod_id": self.pod_id, "principal": self.principal,
            "trace_id": self.trace_id, "kind": kind, "target": target,
            "allowed_topics": list(self.topics),
            "allowed_hosts": list(self.hosts),
            "allowed_ports": list(self.ports),
            "at": time.time()})
        self.recorded_violations += 1

    def check_topic(self, topic: str) -> bool:
        ok = any(topic == t or topic.startswith(t.rstrip("#") ) and
                 (t.endswith("#") or topic == t) for t in self.topics)
        if not ok:
            self._refuse("topic", topic)
        return ok

    def check_host_port(self, host: str, port: int) -> bool:
        ok = host in self.hosts and port in self.ports
        if not ok:
            self._refuse("host_port", f"{host}:{port}")
        return ok

    def stats(self) -> dict[str, Any]:
        return {"pod_id": self.pod_id, "principal": self.principal,
                "violations": self.violations,
                "recorded_violations": self.recorded_violations,
                # Stated, not implied: without a registry a refusal moves a
                # counter in this process and nothing else.
                "violations_logged": self.registry is not None}


@dataclass
class CommResult:
    frames: list[Frame] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    refused: int = 0
    invalid: int = 0
    overflow: int = 0
    recv_data: list[str] = field(default_factory=list)
    #: Set when strict mode refused the output as a whole. The reason is
    #: carried rather than inferred from `invalid`/`overflow`, because a
    #: caller that has to explain a refusal should not have to guess.
    refused_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"executed": self.executed, "refused": self.refused,
                "invalid": self.invalid, "overflow": self.overflow,
                "refused_output": self.refused_output,
                "recv_data": self.recv_data, "frame_count": len(self.frames)}


class NativeCommExecutor:
    """Parses the pod model's output, enforces the egress ACL, and executes
    frames natively: MQTT through a MeshEndpoint, TCP through real sockets."""

    def __init__(self, *, mesh: Any = None, acl: EgressACL,
                 connect_timeout_s: float = 5.0, rate_limit_frames_s: float = 50.0,
                 strict: bool = True):
        self.mesh = mesh
        self.acl = acl
        self.connect_timeout_s = float(connect_timeout_s)
        self.rate_limit_frames_s = float(rate_limit_frames_s)
        self.strict = bool(strict)
        self.refused_outputs = 0
        self._sock: socket.socket | None = None
        self._last_send = 0.0
        self._rate_lock = threading.Lock()

    def execute(self, model_output: str) -> CommResult:
        result = CommResult()
        result.overflow = frame_overflow(model_output)
        frames = parse_frames(model_output)
        result.frames = list(frames)
        result.invalid = sum(1 for frame in frames if not frame.valid)

        if self.strict and (result.invalid or result.overflow):
            # Fail-closed on the OUTPUT, not on the line. A model that
            # produced one line the parser could not read did not produce an
            # instruction that is safe to half-execute, and `sudo rm -rf /`
            # followed by a well-formed PUB used to publish.
            result.refused = len([frame for frame in frames if frame.valid])
            result.refused_output = (
                f"strict mode: {result.invalid} unparsable line(s), "
                f"{result.overflow} frame(s) over the {MAX_FRAMES} limit - "
                "nothing was executed")
            self.refused_outputs += 1
            return result

        dial_host = dial_port = None
        result.frames = []
        result.invalid = 0
        for frame in frames:
            result.frames.append(frame)
            if not frame.valid:
                result.invalid += 1
                continue
            # _rate_ok throttles and always returns True; the refusal branch
            # that used to sit here was unreachable.
            self._rate_ok()
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
        """Throttle, never refuse: a legitimate frame sequence like
        DIAL/SEND/RECV/CLOSE executes within microseconds. Always True —
        the name is kept because callers read it as a guard."""
        with self._rate_lock:
            now = time.monotonic()
            wait = 1.0 / self.rate_limit_frames_s - (now - self._last_send)
            if wait > 0:
                time.sleep(min(wait, 0.5))
            self._last_send = time.monotonic()
            return True

    def stats(self) -> dict[str, Any]:
        return {"strict": self.strict,
                "refused_outputs": self.refused_outputs,
                "acl": self.acl.stats()}

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
