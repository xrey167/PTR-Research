"""Measure PersistentRaftClient framing over a persistent local socket."""
from __future__ import annotations
import json, time
from neural_pods.raft_transport import PersistentRaftClient, PersistentRaftServer, RaftPeer

H = "a" * 64

def _read(c, n):
    out = b""
    while len(out) < n:
        part = c.recv(n - len(out))
        if not part: return b""
        out += part
    return out

def run(count=10000):
    received = [0]
    server = PersistentRaftServer("127.0.0.1", 0, manifest_hash=H,
                                  on_frame=lambda seq, payload, address: received.__setitem__(0, received[0] + 1)).start()
    client = PersistentRaftClient({1: RaftPeer(*server.address)}, manifest_hash=H)
    started = time.perf_counter()
    for _ in range(count): client.send(1, b"x" * 32)
    elapsed = time.perf_counter() - started; client.close(); time.sleep(.02); server.close()
    return {"sent": count, "received": received[0], "ops_s": count / max(elapsed, 1e-9)}

if __name__ == "__main__": print(json.dumps(run(), indent=2))
