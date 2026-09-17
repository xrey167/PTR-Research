from neural_pods.pod_protocol import PodRequest, PodTransport
from neural_pods.pod_socket import PodSocketClient, PodSocketServer, PodSocketSession


def test_socket_transport_roundtrip_and_lineage():
    transport = PodTransport(secret=b"secret")
    transport.register("search", "lookup", lambda payload, request: {"answer": 42}, manifest_hash="m1")
    server = PodSocketServer(transport); host, port = server.start()
    try:
        req = PodRequest("router", "search", "lookup", {}, manifest_hash="m1").sign(b"secret")
        response = PodSocketClient(host, port).dispatch(req)
        assert response.ok and response.payload["answer"] == 42
        assert response.trace_id == req.trace_id and response.request_id == req.request_id
    finally:
        server.close()


def test_persistent_socket_session_reuses_connection():
    transport = PodTransport(secret=b"secret")
    transport.register("search", "lookup", lambda payload, request: {"ok": True}, manifest_hash="m1")
    server = PodSocketServer(transport); host, port = server.start()
    try:
        client = PodSocketClient(host, port); req = PodRequest("router", "search", "lookup", {}, manifest_hash="m1").sign(b"secret")
        with PodSocketSession(client) as session:
            assert session.dispatch(req).ok and session.dispatch(req).ok
    finally:
        server.close()
