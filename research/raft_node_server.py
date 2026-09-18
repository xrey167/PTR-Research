"""Single raft-rs node with real TCP peer transport, meant to run inside an
LXD container (or any host) of the multi-host cluster.

Peer transport uses the same 4-byte length framing as the persistent TCP
benchmark. The control port accepts newline-delimited JSON commands:
  campaign | propose {data} | status | applied | shutdown
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import socket
import socketserver
import struct
import threading
import time

from neural_pods_raft import RaftNode

RAFT_PORT = 45300
CONTROL_PORT = 45400
TICK_INTERVAL_S = 0.05
WAL_PATH = "/tmp/raft-node-{ident}.wal"


def _wal_encode(entries, hard_state) -> str:
    return json.dumps({
        "entries": [[i, t, base64.b64encode(d).decode("ascii")] for i, t, d in entries],
        "hard_state": list(hard_state) if hard_state else None,
    })


def _wal_decode(line: str):
    event = json.loads(line)
    entries = [(i, t, base64.b64decode(d)) for i, t, d in event.get("entries", [])]
    hard_state = tuple(event["hard_state"]) if event.get("hard_state") else None
    return entries, hard_state


class Wal:
    """Crash-recovery write-ahead log: one JSON line per acknowledged Ready."""

    def __init__(self, path: str):
        self.path = path
        self.handle = open(path, "a", encoding="utf-8")

    def append(self, entries, hard_state) -> None:
        self.handle.write(_wal_encode(entries, hard_state) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def load(self) -> tuple[list, list | None]:
        """Fold the WAL into (log entries, latest hard state)."""
        log: dict[int, tuple] = {}
        hard_state = None
        try:
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    entries, hs = _wal_decode(line)
                    for entry in entries:
                        log[entry[0]] = entry
                    if hs is not None:
                        hard_state = hs
        except FileNotFoundError:
            pass
        return sorted(log.values()), hard_state


def read_exact(conn: socket.socket, n: int) -> bytes | None:
    out = b""
    while len(out) < n:
        part = conn.recv(n - len(out))
        if not part:
            return None
        out += part
    return out


class RaftPeerServer:
    PEER_CONN_TTL_S = 30.0

    def __init__(self, ident: int, cluster: list[int], peers: dict[int, str]):
        self.ident = ident
        self.node = RaftNode(ident, cluster)
        self.peers = peers  # ident -> hostname
        self.wal = Wal(WAL_PATH.format(ident=ident))
        entries, hard_state = self.wal.load()
        if entries or hard_state:
            self.node.restore(entries, hard_state)
        self.lock = threading.RLock()
        self.applied: list[bytes] = []
        self.ready_batch: list[tuple] = []
        self.stop = False
        self.out: dict[int, socket.socket] = {}
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", RAFT_PORT))
        self.server.listen(32)
        threading.Thread(target=self.accept_loop, daemon=True).start()
        threading.Thread(target=self.tick_loop, daemon=True).start()

    def accept_loop(self) -> None:
        while not self.stop:
            try:
                self.server.settimeout(0.2)
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self.read_loop, args=(conn,), daemon=True).start()

    def read_loop(self, conn: socket.socket) -> None:
        with conn:
            while not self.stop:
                header = read_exact(conn, 4)
                if header is None:
                    return
                data = read_exact(conn, struct.unpack("!I", header)[0])
                if data is None:
                    return
                with self.lock:
                    try:
                        ready = self.node.step_message(data)
                        if ready is not None:
                            self.ready_batch.append(ready)
                    except Exception:
                        pass  # malformed/stale frames are dropped
                    self.emit()

    def send(self, target: int, data: bytes) -> None:
        entry = self.out.get(target)
        now = time.monotonic()
        if entry is not None and now - entry[1] > self.PEER_CONN_TTL_S:
            entry[0].close()
            entry = None
        conn = entry[0] if entry else None
        if conn is None:
            try:
                conn = socket.create_connection((self.peers[target], RAFT_PORT), timeout=2)
            except OSError:
                return  # peer down; raft retries via progress
            # Detect half-open connections (peer container restarted): without
            # keepalive, sends into a dead socket succeed silently forever.
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            self.out[target] = (conn, now)
        try:
            conn.sendall(struct.pack("!I", len(data)) + data)
        except OSError:
            conn.close()
            self.out.pop(target, None)

    def emit(self) -> None:
        # Raft invariant: persist the Ready (entries + hard state) before it is
        # acknowledged and before its messages are released to peers.
        ready = self.node.poll_ready()
        if ready is not None:
            self.ready_batch.append(ready)
        if not self.ready_batch:
            return
        try:
            for entries, hard_state, _committed in self.ready_batch:
                if entries or hard_state is not None:
                    self.wal.append(entries, hard_state)
            for target, payload in zip(self.node.pending_message_targets(),
                                       self.node.pending_messages()):
                self.send(target, payload)
            committed, released = self.node.ack_ready_messages()
            self.applied.extend(committed)
            for target, payload in released:
                self.send(target, payload)
        except RuntimeError:
            return
        finally:
            self.ready_batch.clear()

    def tick_loop(self) -> None:
        while not self.stop:
            with self.lock:
                try:
                    ready = self.node.tick_pending()
                    if ready is not None:
                        self.ready_batch.append(ready)
                except RuntimeError:
                    pass
                self.emit()
            time.sleep(TICK_INTERVAL_S)

    def status(self) -> dict:
        import re
        term, _commit, soft = self.node.status()
        leader_match = re.search(r"leader_id:\s*(\d+)", soft)
        state_match = re.search(r"raft_state:\s*(\w+)", soft)
        return {"ident": self.ident, "term": term,
                "leader": int(leader_match.group(1)) if leader_match else 0,
                "state": state_match.group(1) if state_match else "",
                "applied": len(self.applied)}

    def close(self) -> None:
        self.stop = True
        for conn, _since in self.out.values():
            conn.close()
        self.server.close()


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for line in self.rfile:
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self.server.peer.lock:
                kind = cmd.get("cmd")
                if kind == "campaign":
                    ready = self.server.peer.node.campaign_pending()
                    if ready is not None:
                        self.server.peer.ready_batch.append(ready)
                    self.server.peer.emit()
                    result = {"ok": True}
                elif kind == "propose":
                    ready = self.server.peer.node.propose_pending(cmd["data"].encode())
                    if ready is not None:
                        self.server.peer.ready_batch.append(ready)
                    self.server.peer.emit()
                    result = {"ok": True}
                elif kind == "status":
                    result = self.server.peer.status()
                elif kind == "shutdown":
                    self.server.peer.close()
                    result = {"ok": True}
                else:
                    result = {"error": "unknown command"}
            self.wfile.write(json.dumps(result).encode() + b"\n")
            if kind == "shutdown":
                return


class ControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ident", type=int, required=True)
    parser.add_argument("--peers", required=True,
                        help="peer spec, e.g. '2=np-node2,3=np-node3'")
    args = parser.parse_args()
    peers = {int(k): v for k, v in (item.split("=", 1) for item in args.peers.split(","))}
    cluster = sorted([args.ident, *peers])
    peer = RaftPeerServer(args.ident, cluster, peers)
    control = ControlServer(("0.0.0.0", CONTROL_PORT), ControlHandler)
    control.peer = peer
    print(json.dumps({"node_ready": args.ident, "cluster": cluster}), flush=True)
    control.serve_forever()


if __name__ == "__main__":
    main()
