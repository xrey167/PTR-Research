"""Client-only benchmark; intentionally runs with the lightweight local Python."""
from __future__ import annotations
import argparse, json, statistics, time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.pod_protocol import PodRequest
from neural_pods.pod_socket import PodSocketClient, PodSocketSession


def run(host, port, requests=1000):
    client=PodSocketClient(host,port,timeout=10); secret=b"cross-host-benchmark"; lat=[]; ok=0
    with PodSocketSession(client) as session:
        for _ in range(requests):
            req=PodRequest("windows-client","research","lookup",{"q":"muller"},manifest_hash="research-v1").sign(secret)
            st=time.perf_counter(); response=session.dispatch(req); lat.append((time.perf_counter()-st)*1000); ok+=int(response.ok)
    lat.sort(); pick=lambda p: lat[min(len(lat)-1,int((len(lat)-1)*p))]
    return {"host":host,"port":port,"requests":requests,"success_rate":ok/requests,"qps":requests/(sum(lat)/1000),"p50_ms":pick(.5),"p95_ms":pick(.95),"p99_ms":pick(.99)}


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("host"); p.add_argument("--port",type=int,default=39123); p.add_argument("--requests",type=int,default=1000); a=p.parse_args(); print(json.dumps(run(a.host,a.port,a.requests),indent=2))
