"""Persistent, authenticated framing for Raft protobuf messages.

The Raft binding owns consensus; this module owns peer transport.  mTLS
authenticates the channel and the manifest digest binds a frame to the exact
Pod/cluster generation.  Connections are reused and retries are bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
import socket, ssl, struct, threading
from typing import Mapping

_MAGIC = b"NPR1"
_HEADER = struct.Struct("!4sIQ")  # magic, frame bytes, sequence


class RaftFrameError(ValueError):
    pass


def encode_raft_frame(payload: bytes, *, manifest_hash: str, sequence: int = 0,
                      max_bytes: int = 16 * 1024 * 1024) -> bytes:
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    digest = manifest_hash.encode("ascii")
    if len(digest) != 64:
        raise ValueError("manifest_hash must be a SHA-256 hex digest")
    body = digest + bytes(payload)
    if len(body) > max_bytes:
        raise ValueError("Raft frame exceeds max_bytes")
    return _HEADER.pack(_MAGIC, len(body), int(sequence)) + body


def decode_raft_frame(frame: bytes, *, expected_manifest: str,
                      max_bytes: int = 16 * 1024 * 1024) -> tuple[int, bytes]:
    if len(frame) < _HEADER.size:
        raise RaftFrameError("truncated header")
    magic, size, sequence = _HEADER.unpack(frame[:_HEADER.size])
    if magic != _MAGIC or size > max_bytes or len(frame) != _HEADER.size + size:
        raise RaftFrameError("invalid frame")
    body = frame[_HEADER.size:]
    if body[:64].decode("ascii", errors="ignore") != expected_manifest:
        raise RaftFrameError("manifest mismatch")
    return sequence, body[64:]


@dataclass(frozen=True)
class RaftPeer:
    host: str
    port: int


class PersistentRaftClient:
    def __init__(self, peers: Mapping[int, RaftPeer], *, manifest_hash: str,
                 ssl_context: ssl.SSLContext | None = None, timeout: float = 2.0,
                 max_retries: int = 2, max_frame_bytes: int = 16 * 1024 * 1024):
        if max_retries < 0: raise ValueError("max_retries must be non-negative")
        self.peers, self.manifest_hash, self.ssl_context = dict(peers), manifest_hash, ssl_context
        self.timeout, self.max_retries, self.max_frame_bytes = timeout, max_retries, max_frame_bytes
        self._sockets: dict[int, socket.socket] = {}; self._sequence = 0; self._lock = threading.RLock()

    def _connect(self, peer_id: int) -> socket.socket:
        peer = self.peers[peer_id]
        raw = socket.create_connection((peer.host, peer.port), timeout=self.timeout)
        if self.ssl_context is not None:
            raw = self.ssl_context.wrap_socket(raw, server_hostname=peer.host)
        raw.settimeout(self.timeout); return raw

    def send(self, peer_id: int, payload: bytes) -> int:
        frame = None
        with self._lock:
            self._sequence += 1
            frame = encode_raft_frame(payload, manifest_hash=self.manifest_hash,
                                      sequence=self._sequence, max_bytes=self.max_frame_bytes)
            last: Exception | None = None
            for _ in range(self.max_retries + 1):
                try:
                    sock = self._sockets.get(peer_id)
                    if sock is None:
                        sock = self._connect(peer_id); self._sockets[peer_id] = sock
                    sock.sendall(frame); return self._sequence
                except (OSError, ssl.SSLError) as exc:
                    last = exc; old = self._sockets.pop(peer_id, None)
                    if old is not None: old.close()
            raise ConnectionError(f"peer {peer_id} unavailable after retries") from last

    def close(self) -> None:
        with self._lock:
            for sock in self._sockets.values(): sock.close()
            self._sockets.clear()


class PersistentRaftServer:
    """Bounded frame server for a Raft peer.

    The callback runs only after the manifest and frame size have been
    validated. It receives `(sequence, payload, peer_address)`.
    """
    def __init__(self, host: str, port: int, *, manifest_hash: str,
                 on_frame, ssl_context: ssl.SSLContext | None = None,
                 max_frame_bytes: int = 16 * 1024 * 1024,
                 allowed_peer_subjects: set[str] | None = None):
        self.host, self.port, self.manifest_hash = host, int(port), manifest_hash
        self.on_frame, self.ssl_context, self.max_frame_bytes = on_frame, ssl_context, max_frame_bytes
        self.allowed_peer_subjects = set(allowed_peer_subjects or ())
        self._stop = threading.Event(); self._threads: list[threading.Thread] = []
        self._server = socket.socket(); self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, int(port))); self._server.listen(64); self.address = self._server.getsockname()
        self._accept_thread: threading.Thread | None = None

    @staticmethod
    def _read_exact(sock: socket.socket, size: int) -> bytes | None:
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk: return None
            data.extend(chunk)
        return bytes(data)

    def start(self) -> "PersistentRaftServer":
        self._accept_thread = threading.Thread(target=self._accept, daemon=True); self._accept_thread.start()
        return self

    def _accept(self) -> None:
        self._server.settimeout(.2)
        while not self._stop.is_set():
            try: raw, address = self._server.accept()
            except socket.timeout: continue
            except OSError: break
            try: sock = self.ssl_context.wrap_socket(raw, server_side=True) if self.ssl_context else raw
            except ssl.SSLError: raw.close(); continue
            if self.allowed_peer_subjects:
                cert = sock.getpeercert()
                subject = {key: value for part in cert.get("subject", ()) for key, value in part}
                common_name = subject.get("commonName")
                if common_name not in self.allowed_peer_subjects:
                    sock.close(); continue
            thread = threading.Thread(target=self._connection, args=(sock, address), daemon=True)
            self._threads.append(thread); thread.start()

    def _connection(self, sock: socket.socket, address) -> None:
        with sock:
            while not self._stop.is_set():
                header = self._read_exact(sock, _HEADER.size)
                if header is None: return
                try: _, size, _ = _HEADER.unpack(header)
                except struct.error: return
                body = self._read_exact(sock, size)
                if body is None: return
                try:
                    sequence, payload = decode_raft_frame(header + body,
                        expected_manifest=self.manifest_hash, max_bytes=self.max_frame_bytes)
                except (RaftFrameError, UnicodeError, ValueError):
                    return
                self.on_frame(sequence, payload, address)

    def close(self) -> None:
        self._stop.set(); self._server.close()
        if self._accept_thread: self._accept_thread.join(timeout=1)
