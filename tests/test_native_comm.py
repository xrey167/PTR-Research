import base64

from neural_pods.native_comm import (EgressACL, Frame, NativeCommExecutor,
                                     parse_frames)


def test_parser_accepts_valid_dialect():
    text = ("PUB np/reader/answer {\"value\": 42}\n"
            "DIAL 10.50.0.121:45300\n"
            "SEND " + base64.b64encode(b"hello").decode() + "\n"
            "RECV\n"
            "CLOSE\n"
            "SUB np/broadcast/#")
    frames = parse_frames(text)
    assert [f.kind for f in frames] == ["PUB", "DIAL", "SEND", "RECV", "CLOSE", "SUB"]
    assert all(f.valid for f in frames)


def test_parser_fail_closed_on_garbage():
    frames = parse_frames("sudo rm -rf /\n"
                          "PUB not-a-topic {}\n"
                          "PUB np/ok {not json}\n"
                          "SEND !!!not-base64!!!\n"
                          "DIAL no-port\n")
    assert all(not f.valid for f in frames)
    assert len(frames) == 5


def test_parser_caps_frame_count():
    frames = parse_frames("\n".join(["RECV"] * 100))
    assert len(frames) == 16  # MAX_FRAMES


def test_acl_blocks_forbidden_targets():
    acl = EgressACL(topics=("np/reader/answer",), hosts=("10.50.0.121",), ports=(45300,))
    assert acl.check_topic("np/reader/answer")
    assert not acl.check_topic("np/secret/leak")
    assert acl.check_host_port("10.50.0.121", 45300)
    assert not acl.check_host_port("evil.example.com", 4444)
    assert acl.violations == 2


def test_executor_executes_and_refuses():
    published = {}

    class FakeMesh:
        def publish_raw(self, topic, body, *, retain=False):
            published[topic] = body

    acl = EgressACL(topics=("np/reader/answer",), hosts=("127.0.0.1",), ports=(45999,))
    executor = NativeCommExecutor(mesh=FakeMesh(), acl=acl)
    text = ("PUB np/reader/answer {\"value\": 42}\n"
            "PUB np/forbidden/x {\"steal\": true}\n"
            "TOTALLY INVALID LINE\n")
    result = executor.execute(text)
    assert published == {"np/reader/answer": {"value": 42}}
    assert result.refused == 1
    assert result.invalid == 1
    assert result.to_dict()["executed"][0]["topic"] == "np/reader/answer"


def test_executor_tcp_dial_send_recv_close():
    import socket as socket_mod
    import threading

    server = socket_mod.socket()
    server.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)
    received = []

    def serve():
        conn, _ = server.accept()
        received.append(conn.recv(1024))
        conn.sendall(b"pong")
        conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()

    acl = EgressACL(topics=("np/x",), hosts=("127.0.0.1",), ports=(port,))
    executor = NativeCommExecutor(mesh=None, acl=acl)
    payload = base64.b64encode(b"ping").decode()
    text = (f"DIAL 127.0.0.1:{port}\n"
            f"SEND {payload}\n"
            "RECV\n"
            "CLOSE\n")
    result = executor.execute(text)
    assert received == [b"ping"]
    assert result.recv_data == ["pong"]
    assert [f.kind for f in result.frames] == ["DIAL", "SEND", "RECV", "CLOSE"]
    assert result.refused == 0 and result.invalid == 0
