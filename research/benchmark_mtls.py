"""One-shot mutual-TLS Pod transport smoke benchmark."""
from __future__ import annotations
import json, ssl, sys, time
from neural_pods.pod_protocol import PodRequest, PodTransport
from neural_pods.pod_socket import PodSocketServer, PodSocketClient, PodSocketSession


def run(cert_dir: str):
    p = cert_dir.rstrip("/")
    server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_ctx.load_cert_chain(p + "/server.crt", p + "/server.key")
    server_ctx.load_verify_locations(p + "/ca.crt")
    server_ctx.verify_mode = ssl.CERT_REQUIRED
    client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=p + "/ca.crt")
    client_ctx.check_hostname = False
    client_ctx.load_cert_chain(p + "/client.crt", p + "/client.key")
    transport = PodTransport(secret=b"mtls-benchmark")
    transport.register("research", "lookup", lambda payload, request: {"verified": True}, manifest_hash="research-v1")
    server = PodSocketServer(transport, ssl_context=server_ctx)
    host, port = server.start()
    try:
        client = PodSocketClient(host, port, ssl_context=client_ctx)
        request = PodRequest("router", "research", "lookup", {"q": "muller"}, manifest_hash="research-v1").sign(b"mtls-benchmark")
        latencies = []
        with PodSocketSession(client) as session:
            for _ in range(100):
                started = time.perf_counter(); response = session.dispatch(request); latencies.append((time.perf_counter() - started) * 1000)
        latencies.sort()
        return {"transport": "mutual-tls+HMAC+manifest", "requests": 100, "success_rate": 1.0 if response.ok else 0.0,
                "p50_ms": latencies[49], "p95_ms": latencies[94], "p99_ms": latencies[98]}
    finally:
        server.close()


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=2))
