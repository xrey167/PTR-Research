"""Single raft-rs node with real TCP peer transport, meant to run inside an
LXD container (or any host) of the multi-host cluster.

Peer transport uses the same 4-byte length framing as the persistent TCP
benchmark. The control port accepts newline-delimited JSON commands:
  campaign | propose {data} | status | applied | shutdown
"""
from __future__ import annotations
import argparse
import json
import socket
import socketserver
import struct
import threading
import time

from neural_pods_raft import RaftNode

RAFT_PORT = 45300
CONTROL_PORT = 45400
TICK_INTERVAL_S = 0.05


def read_exact(conn: socket.socket, n: int) -> bytes | None:
    out = b""
    while len(out) < n:
        part = conn.recv(n - len(out))
        if not part:
            return None
        out += part
    return out


class RaftPeerServer:
    def __init__(self, ident: int, cluster: list[int], peers: dict[int, str]):
        self.ident = ident
        self.node = RaftNode(ident, cluster)
        self.peers = peers  # ident -> hostname
        self.lock = threading.RLock()
        self.applied: list[bytes] = []
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
                        if self.node.step_message(data) is not None:
                            self.emit()
                    except Exception:
                        pass  # malformed/stale frames are dropped

    def send(self, target: int, data: bytes) -> None:
        conn = self.out.get(target)
        if conn is None:
            try:
                conn = socket.create_connection((self.peers[target], RAFT_PORT), timeout=2)
            except OSError:
                return  # peer down; raft retries via progress
            self.out[target] = conn
        try:
            conn.sendall(struct.pack("!I", len(data)) + data)
        except OSError:
            conn.close()
            self.out.pop(target, None)

    def emit(self) -> None:
        for target, payload in zip(self.node.pending_message_targets(), self.node.pending_messages()):
            self.send(target, payload)
        self.applied.extend(self.node.ack_ready())

    def tick_loop(self) -> None:
        while not self.stop:
            with self.lock:
                self.node.tick()
                self.emit()
            time.sleep(TICK_INTERVAL_S)

    def status(self) -> dict:
        term, _commit, soft = self.node.status()
        leader = 0
        state = ""
        for part in soft.split(";"):
            part = part.strip()
            if part.startswith("raft_state:"):
                state = part.split(":", 1)[1].strip().rstrip(" }")
            if part.startswith("leader_id:"):
                leader = int(part.split(":", 1)[1].strip().rstrip(" }"))
        return {"ident": self.ident, "term": term, "leader": leader, "state": state,
                "applied": len(self.applied)}

    def close(self) -> None:
        self.stop = True
        for conn in self.out.values():
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
                    self.server.peer.node.campaign_pending()
                    self.server.peer.emit()
                    result = {"ok": True}
                elif kind == "propose":
                    self.server.peer.node.propose_pending(cmd["data"].encode())
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
