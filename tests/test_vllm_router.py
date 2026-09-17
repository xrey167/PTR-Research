import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from neural_pods.vllm_router import VllmReplica, VllmReplicaRouter
from unittest.mock import patch


class _Handler(BaseHTTPRequestHandler):
    healthy = True
    calls = 0

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(200 if self.healthy else 503)
        self.end_headers()

    def do_POST(self):
        type(self).calls += 1
        n = int(self.headers.get("content-length", 0))
        json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "replica": self.server.replica_name}).encode())


def _server(name):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.replica_name = name
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_router_round_robin_and_health():
    a, b = _server("a"), _server("b")
    try:
        router = VllmReplicaRouter([
            VllmReplica("a", f"http://127.0.0.1:{a.server_port}"),
            VllmReplica("b", f"http://127.0.0.1:{b.server_port}"),
        ])
        assert all(router.health().values())
        assert [router.chat({"x": i})["replica"] for i in range(4)] == ["a", "b", "a", "b"]
        assert router.metrics.successes == 4
        assert router.metrics.failures == 0
    finally:
        a.shutdown(); b.shutdown()


def test_router_bounded_failover():
    dead = VllmReplica("dead", "http://127.0.0.1:1")
    live = _server("live")
    try:
        router = VllmReplicaRouter([dead, VllmReplica("live", f"http://127.0.0.1:{live.server_port}")], timeout_s=.1)
        result = router.chat({"x": 1})
        assert result["replica"] == "live"
        assert router.metrics.failovers == 1
        assert router.metrics.per_replica["dead"]["failures"] == 1
    finally:
        live.shutdown()


def test_router_passes_mtls_context_to_https_calls():
    context = ssl.create_default_context()
    replica = VllmReplica("secure", "https://example.invalid")
    router = VllmReplicaRouter([replica], ssl_context=context)

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return b'{"ok": true}'

    seen = []
    def fake_open(request, timeout, context):
        seen.append((getattr(request, "full_url", request), timeout, context))
        return Response()

    with patch("urllib.request.urlopen", side_effect=fake_open):
        assert router.health()["secure"]
        assert router.chat({"x": 1})["ok"]
    assert len(seen) == 2
    assert all(item[2] is context for item in seen)
