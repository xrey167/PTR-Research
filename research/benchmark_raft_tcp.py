"""TCP loopback proof for the PyO3 Raft message boundary."""
from __future__ import annotations
import json, socket, struct, threading, time
from neural_pods_raft import RaftNode

BASE = 45200


class Peer:
    def __init__(self, node_id: int):
        self.node_id, self.port = node_id, BASE + node_id
        self.node = RaftNode(node_id, [1, 2, 3]); self.lock = threading.RLock()
        self.applied: list[bytes] = []; self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", self.port)); self.server.listen(32); self.stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while not self.stop:
            try:
                self.server.settimeout(.2); conn, _ = self.server.accept()
            except socket.timeout:
                continue
            with conn:
                header = conn.recv(4)
                if not header: continue
                size = struct.unpack("!I", header)[0]; data = b""
                while len(data) < size: data += conn.recv(size - len(data))
            with self.lock:
                if self.node.step_message(data) is not None: self.emit()

    def emit(self):
        targets = self.node.pending_message_targets(); messages = self.node.pending_messages()
        for target, data in zip(targets, messages):
            with socket.create_connection(("127.0.0.1", BASE + target), timeout=2) as conn:
                conn.sendall(struct.pack("!I", len(data)) + data)
        self.applied.extend(self.node.ack_ready())

    def close(self): self.stop = True; self.server.close()


def run(count: int = 20) -> dict:
    peers = {i: Peer(i) for i in (1, 2, 3)}; time.sleep(.02)
    try:
        with peers[1].lock: peers[1].node.campaign_pending(); peers[1].emit()
        deadline = time.time() + 3
        while time.time() < deadline and "Leader" not in peers[1].node.status()[2]: time.sleep(.001)
        started = time.perf_counter()
        for i in range(count):
            with peers[1].lock: peers[1].node.propose_pending(f"tcp-{i}".encode()); peers[1].emit()
        while time.time() < deadline and min(len(p.applied) for p in peers.values()) < count: time.sleep(.001)
        elapsed = time.perf_counter() - started
        return {"nodes": 3, "proposals": count, "elapsed_s": elapsed,
                "proposals_per_s": count / max(elapsed, 1e-9),
                "applied": {str(i): len(p.applied) for i, p in peers.items()},
                "leader": peers[1].node.status()}
    finally:
        for p in peers.values(): p.close()


if __name__ == "__main__": print(json.dumps(run(), indent=2, default=str))
