"""Persistent TCP comparison for the native Raft protobuf boundary."""
from __future__ import annotations
import json, socket, struct, threading, time
from neural_pods_raft import RaftNode

BASE = 45300

def read_exact(conn, n):
    out = b""
    while len(out) < n:
        part = conn.recv(n - len(out))
        if not part: return None
        out += part
    return out

class Peer:
    def __init__(self, ident):
        self.ident, self.port = ident, BASE + ident
        self.node, self.lock = RaftNode(ident, [1,2,3]), threading.RLock()
        self.applied, self.stop, self.out = [], False, {}
        self.server = socket.socket(); self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", self.port)); self.server.listen(32)
        threading.Thread(target=self.accept, daemon=True).start()
    def accept(self):
        while not self.stop:
            try: self.server.settimeout(.2); c,_=self.server.accept()
            except socket.timeout: continue
            threading.Thread(target=self.read_loop,args=(c,),daemon=True).start()
    def read_loop(self,c):
        with c:
            while not self.stop:
                h=read_exact(c,4)
                if h is None: return
                data=read_exact(c,struct.unpack('!I',h)[0])
                if data is None: return
                with self.lock:
                    if self.node.step_message(data) is not None: self.emit()
    def send(self,target,data):
        c=self.out.get(target)
        if c is None:
            c=socket.create_connection(('127.0.0.1',BASE+target),timeout=2); self.out[target]=c
        c.sendall(struct.pack('!I',len(data))+data)
    def emit(self):
        for target,data in zip(self.node.pending_message_targets(),self.node.pending_messages()): self.send(target,data)
        self.applied.extend(self.node.ack_ready())
    def close(self):
        self.stop=True
        for c in self.out.values(): c.close()
        self.server.close()

def run(count=100):
    ps={i:Peer(i) for i in (1,2,3)}; time.sleep(.02)
    try:
        with ps[1].lock: ps[1].node.campaign_pending(); ps[1].emit()
        deadline=time.time()+5
        while time.time()<deadline and 'Leader' not in ps[1].node.status()[2]: time.sleep(.001)
        start=time.perf_counter()
        for i in range(count):
            with ps[1].lock: ps[1].node.propose_pending(f'p-{i}'.encode()); ps[1].emit()
        while time.time()<deadline and min(len(p.applied) for p in ps.values())<count: time.sleep(.001)
        elapsed=time.perf_counter()-start
        return {'proposals':count,'elapsed_s':elapsed,'proposals_per_s':count/max(elapsed,1e-9),'applied':{str(i):len(p.applied) for i,p in ps.items()}}
    finally:
        for p in ps.values(): p.close()

if __name__=='__main__': print(json.dumps(run(),indent=2))
