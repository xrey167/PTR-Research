from __future__ import annotations
import json, statistics, time
from neural_pods.replication import QuorumReplicator


class MemoryReplica:
    def __init__(self): self.data={}; self.failed=False
    def put(self,k,v):
        if self.failed: raise ConnectionError("offline")
        self.data[k]=dict(v)
    def get(self,k):
        if self.failed: raise ConnectionError("offline")
        return self.data.get(k)


def run(writes=1000):
    replicas={x:MemoryReplica() for x in ("a","b","c")}; q=QuorumReplicator(replicas,quorum=2); lat=[]
    for i in range(writes):
        st=time.perf_counter(); q.write("pod:demo",{"value":i}); lat.append((time.perf_counter()-st)*1000)
    replicas["c"].failed=True; st=time.perf_counter(); q.write("pod:demo",{"value":writes}); degraded=(time.perf_counter()-st)*1000
    latest=q.read("pod:demo")
    lat.sort(); pick=lambda p:lat[min(len(lat)-1,int((len(lat)-1)*p))]
    return {"writes":writes,"healthy_p50_ms":pick(.5),"healthy_p95_ms":pick(.95),"degraded_write_ms":degraded,"latest":latest["payload"]["value"],"healthy_replicas":2,"quorum":2}


if __name__=="__main__": print(json.dumps(run(),indent=2))
