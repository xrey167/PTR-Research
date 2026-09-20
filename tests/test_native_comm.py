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


class FakeMesh:
    def __init__(self):
        self.published = {}

    def publish_raw(self, topic, body, *, retain=False):
        self.published[topic] = body

    def subscribe(self, topic, handler):
        pass


def test_executor_executes_and_refuses_per_frame_when_not_strict():
    mesh = FakeMesh()
    acl = EgressACL(topics=("np/reader/answer",), hosts=("127.0.0.1",), ports=(45999,))
    executor = NativeCommExecutor(mesh=mesh, acl=acl, strict=False)
    text = ("PUB np/reader/answer {\"value\": 42}\n"
            "PUB np/forbidden/x {\"steal\": true}\n"
            "TOTALLY INVALID LINE\n")
    result = executor.execute(text)
    assert mesh.published == {"np/reader/answer": {"value": 42}}
    assert result.refused == 1
    assert result.invalid == 1
    assert result.refused_output is None
    assert result.to_dict()["executed"][0]["topic"] == "np/reader/answer"


def test_strict_mode_refuses_the_whole_output_when_one_line_is_unparsable():
    """Per-frame refusal is not fail-closed. An output with a line the parser
    could not read is an output that was not understood, and the valid lines
    around it used to execute anyway — `sudo rm -rf /` followed by a
    well-formed PUB published."""
    mesh = FakeMesh()
    acl = EgressACL(topics=("np/reader/answer",))
    executor = NativeCommExecutor(mesh=mesh, acl=acl)          # strict by default
    text = ("sudo rm -rf /\n"
            "PUB np/reader/answer {\"value\": 42}\n")
    result = executor.execute(text)

    assert mesh.published == {}
    assert result.executed == []
    assert result.invalid == 1
    assert result.refused == 1                 # the frame that did not run
    assert "strict mode" in result.refused_output
    assert executor.stats()["refused_outputs"] == 1


def test_strict_mode_executes_an_output_it_fully_understands():
    mesh = FakeMesh()
    acl = EgressACL(topics=("np/reader/answer",))
    executor = NativeCommExecutor(mesh=mesh, acl=acl)
    result = executor.execute('PUB np/reader/answer {"value": 42}\n')

    assert mesh.published == {"np/reader/answer": {"value": 42}}
    assert result.refused_output is None
    assert result.invalid == 0
    assert executor.stats()["refused_outputs"] == 0


def test_strict_mode_refuses_an_output_that_overflows_the_frame_limit():
    from neural_pods.native_comm import MAX_FRAMES

    mesh = FakeMesh()
    executor = NativeCommExecutor(mesh=mesh, acl=EgressACL(topics=("np/x/y",)))
    text = "\n".join(f'PUB np/x/y {{"i":{i}}}' for i in range(MAX_FRAMES + 5))
    result = executor.execute(text)

    assert mesh.published == {}
    assert result.overflow == 5
    assert "over the" in result.refused_output


def test_acl_refusals_are_written_to_the_provenance_log():
    """A refused frame is a pod addressing what it may not. Counting it in
    process memory and nowhere else made it unreviewable afterwards."""
    from neural_pods.registry import Registry

    registry = Registry(":memory:")
    acl = EgressACL(topics=("np/reader/answer",), registry=registry,
                    pod_id="reader-gen6", trace_id="trace-1")
    executor = NativeCommExecutor(mesh=FakeMesh(), acl=acl, strict=False)
    executor.execute('PUB np/forbidden/x {"steal": true}\n')

    events = registry.events(action="egress_refused")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pod_id"] == "reader-gen6"
    assert payload["target"] == "np/forbidden/x"
    assert payload["trace_id"] == "trace-1"
    assert acl.stats() == {"pod_id": "reader-gen6", "principal": "local",
                           "violations": 1, "recorded_violations": 1,
                           "violations_logged": True}


def test_without_a_registry_the_acl_says_refusals_are_not_logged():
    acl = EgressACL(topics=("np/reader/answer",), pod_id="reader-gen6")
    assert acl.check_topic("np/forbidden/x") is False
    stats = acl.stats()
    assert stats["violations"] == 1
    assert stats["recorded_violations"] == 0
    assert stats["violations_logged"] is False


def test_the_egress_acl_is_derived_from_the_link_contract():
    """The module docstring said the ACL came from the link contract while
    PodLink carried no egress fields and every ACL was hand-built."""
    from neural_pods.pod_contract import PodLink

    link = PodLink(trace_id="t-1", source_pod_id="reader-gen6",
                   target_pod_id="lookup", target_generation="g7",
                   artifact_id="a-1", transport="mqtt", capability="lookup",
                   acl=("tenant-a",),
                   egress_topics=("np/reader/answer",),
                   egress_hosts=("10.50.0.121",), egress_ports=(45300,))
    acl = EgressACL.from_link(link)

    assert acl.topics == ("np/reader/answer",)
    assert acl.check_host_port("10.50.0.121", 45300) is True
    assert acl.check_topic("np/secret/leak") is False
    assert acl.principal == "tenant-a"
    assert acl.pod_id == "reader-gen6"
    assert acl.trace_id == "t-1"


def test_a_link_that_grants_no_egress_yields_an_acl_that_permits_nothing():
    from neural_pods.pod_contract import PodLink

    link = PodLink(trace_id="t-2", source_pod_id="p", target_pod_id="q",
                   target_generation="g", artifact_id="a", transport="mqtt",
                   capability="c", acl=("*",))
    acl = EgressACL.from_link(link)
    assert acl.check_topic("np/anything") is False
    assert acl.check_host_port("127.0.0.1", 80) is False


def test_the_next_hop_carries_the_egress_allowlist():
    """A hop that dropped it would hand the next pod a contract granting
    nothing while the code that reads it would see an empty allowlist —
    or, worse, a caller would rebuild an open one."""
    from neural_pods.pod_contract import PodLink

    link = PodLink(trace_id="t-3", source_pod_id="p", target_pod_id="q",
                   target_generation="g", artifact_id="a", transport="mqtt",
                   capability="c", acl=("*",),
                   egress_topics=("np/a/b",), egress_hosts=("h",),
                   egress_ports=(1,))
    hop = link.next_hop()
    assert hop.egress_topics == ("np/a/b",)
    assert hop.egress_hosts == ("h",)
    assert hop.egress_ports == (1,)
    assert hop.hop_budget == link.hop_budget - 1


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


def test_frames_beyond_the_limit_are_counted_not_silently_dropped():
    """parse_frames slices at MAX_FRAMES. A fail-closed parser that drops
    input without saying so leaves the caller believing everything ran."""
    from neural_pods.native_comm import (MAX_FRAMES, EgressACL,
                                         NativeCommExecutor, frame_overflow)

    text = "\n".join(f'PUB np/x/y {{"i":{i}}}' for i in range(MAX_FRAMES + 5))
    assert frame_overflow(text) == 5

    executor = NativeCommExecutor(mesh=None, acl=EgressACL(topics=()))
    result = executor.execute(text)
    assert result.overflow == 5
    assert len(result.frames) == MAX_FRAMES
    assert result.to_dict()["overflow"] == 5


def test_no_overflow_is_reported_for_a_normal_frame_sequence():
    from neural_pods.native_comm import EgressACL, NativeCommExecutor

    executor = NativeCommExecutor(mesh=None, acl=EgressACL(topics=()))
    result = executor.execute('PUB np/a/b {"x":1}\nSUB np/a/c\n')
    assert result.overflow == 0
