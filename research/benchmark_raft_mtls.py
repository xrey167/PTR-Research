"""Persistent mutual-TLS Raft peer benchmark."""
from __future__ import annotations
import json, socket, ssl, struct, threading, time
from neural_pods_raft import RaftNode

BASE, CERT = 45400, "/tmp/raftcert"

def exact(c, n):
    b=b""
    while len(b)<n:
        x=c.recv(n-len(b))
        if not x: return None
        b+=x
    return b

class Peer:
    def __init__(self, ident):
        self.ident,self.port=ident,BASE+ident; self.node=RaftNode(ident,[1,2,3]); self.lock=threading.RLock()
        self.applied=[]; self.out={}; self.stop=False
        self.server_ctx=ssl.create_default_context(ssl.Purpose.CLIENT_AUTH,cafile=CERT+'/ca.crt')
        self.server_ctx.load_cert_chain(CERT+f'/n{ident}.crt',CERT+f'/n{ident}.key'); self.server_ctx.verify_mode=ssl.CERT_REQUIRED
        self.client_ctx=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=CERT+'/ca.crt'); self.client_ctx.check_hostname=False
        self.client_ctx.load_cert_chain(CERT+f'/n{ident}.crt',CERT+f'/n{ident}.key')
        self.sock=socket.socket(); self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); self.sock.bind(('127.0.0.1',self.port)); self.sock.listen(32)
        threading.Thread(target=self.accept,daemon=True).start()
    def accept(self):
        while not self.stop:
            try: self.sock.settimeout(.2); raw,_=self.sock.accept(); c=self.server_ctx.wrap_socket(raw,server_side=True)
            except socket.timeout: continue
            threading.Thread(target=self.read,args=(c,),daemon=True).start()
    def read(self,c):
        with c:
            while not self.stop:
                h=exact(c,4)
                if h is None:return
                data=exact(c,struct.unpack('!I',h)[0])
                if data is None:return
                with self.lock:
                    if self.node.step_message(data) is not None:self.emit()
    def send(self,target,data):
        c=self.out.get(target)
        if c is None:
            raw=socket.create_connection(('127.0.0.1',BASE+target),timeout=2); c=self.client_ctx.wrap_socket(raw,server_hostname='raft'); self.out[target]=c
        c.sendall(struct.pack('!I',len(data))+data)
    def emit(self):
        for target,data in zip(self.node.pending_message_targets(),self.node.pending_messages()):self.send(target,data)
        self.applied.extend(self.node.ack_ready())
    def close(self):
        self.stop=True
        for c in self.out.values():c.close()
        self.sock.close()

def run(count=100):
    ps={i:Peer(i) for i in (1,2,3)}; time.sleep(.05)
    try:
        with ps[1].lock:ps[1].node.campaign_pending();ps[1].emit()
        deadline=time.time()+5
        while time.time()<deadline and 'Leader' not in ps[1].node.status()[2]:time.sleep(.001)
        start=time.perf_counter()
        for i in range(count):
            with ps[1].lock:ps[1].node.propose_pending(f'mtls-{i}'.encode());ps[1].emit()
        while time.time()<deadline and min(len(p.applied) for p in ps.values())<count:time.sleep(.001)
        elapsed=time.perf_counter()-start
        return {'proposals':count,'elapsed_s':elapsed,'proposals_per_s':count/max(elapsed,1e-9),'applied':{str(i):len(p.applied) for i,p in ps.items()}}
    finally:
        for p in ps.values():p.close()

if __name__=='__main__':print(json.dumps(run(),indent=2))
