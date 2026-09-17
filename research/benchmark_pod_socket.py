"""Network burst benchmark for the Pod protocol on one host."""
from __future__ import annotations
import argparse, json, statistics, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from neural_pods.pod_protocol import PodRequest, PodTransport
from neural_pods.pod_socket import PodSocketServer, PodSocketClient


def percentile(xs, p):
    ys = sorted(xs); return ys[min(len(ys)-1, int((len(ys)-1)*p))]


def _process_batch(args):
    host, port, secret, requests = args
    client = PodSocketClient(host, port, timeout=10); ok = 0; latencies = []
    for _ in range(requests):
        req = PodRequest("router", "research", "lookup", {"q": "muller"}, manifest_hash="research-v1").sign(secret)
        started = time.perf_counter(); response = client.dispatch(req); latencies.append((time.perf_counter()-started)*1000); ok += int(response.ok)
    return latencies, ok


def run(requests=2000, workers=8, processes=1, host="127.0.0.1"):
    secret = b"benchmark-secret"; transport = PodTransport(secret=secret)
    transport.register("research", "lookup", lambda payload, req: {"found": True}, manifest_hash="research-v1")
    server = PodSocketServer(transport, host=host); host, port = server.start()
    template = PodRequest("router", "research", "lookup", {"q": "muller"}, manifest_hash="research-v1").sign(secret)
    def one(_):
        client = PodSocketClient(host, port, timeout=5); req = PodRequest(**{**template.__dict__, "request_id": PodRequest("a","b","c",{}).request_id}).sign(secret)
        started = time.perf_counter(); response = client.dispatch(req)
        return (time.perf_counter()-started)*1000, response.ok
    started = time.perf_counter()
    if processes > 1:
        batches = [requests // processes + int(i < requests % processes) for i in range(processes)]
        with Pool(processes) as pool: grouped = pool.map(_process_batch, [(host, port, secret, n) for n in batches if n])
        latencies = [latency for group, _ in grouped for latency in group]; successes = sum(ok for _, ok in grouped)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool: results = list(pool.map(one, range(requests)))
        latencies = [x[0] for x in results]; successes = sum(x[1] for x in results)
    elapsed = time.perf_counter()-started; server.close()
    return {"requests": requests, "workers": workers, "processes": processes, "qps": requests/elapsed,
            "latency_ms": {"p50": percentile(latencies,.50), "p95": percentile(latencies,.95), "p99": percentile(latencies,.99)},
            "success_rate": successes/len(latencies), "transport": "threaded-tcp-jsonl"}


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--requests",type=int,default=2000); p.add_argument("--workers",type=int,default=8); p.add_argument("--processes",type=int,default=1); p.add_argument("--host",default="127.0.0.1"); a=p.parse_args()
    print(json.dumps(run(a.requests,a.workers,a.processes,a.host), indent=2))
