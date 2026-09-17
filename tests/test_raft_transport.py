import pytest
from neural_pods.raft_transport import (RaftFrameError, decode_raft_frame,
                                         encode_raft_frame, PersistentRaftClient,
                                         PersistentRaftServer, RaftPeer)
import time


HASH = "a" * 64


def test_raft_frame_roundtrip_and_manifest_fence():
    frame = encode_raft_frame(b"raft", manifest_hash=HASH, sequence=7)
    assert decode_raft_frame(frame, expected_manifest=HASH) == (7, b"raft")
    with pytest.raises(RaftFrameError, match="manifest"):
        decode_raft_frame(frame, expected_manifest="b" * 64)


def test_raft_frame_size_is_bounded():
    with pytest.raises(ValueError, match="max_bytes"):
        encode_raft_frame(b"x" * 100, manifest_hash=HASH, max_bytes=64)


def test_persistent_client_server_roundtrip():
    seen = []
    server = PersistentRaftServer("127.0.0.1", 0, manifest_hash=HASH,
                                  on_frame=lambda seq, payload, addr: seen.append((seq, payload)))
    server.start()
    client = PersistentRaftClient({1: RaftPeer(*server.address)}, manifest_hash=HASH)
    try:
        client.send(1, b"hello")
        deadline = time.time() + 1
        while time.time() < deadline and not seen: time.sleep(.001)
        assert seen == [(1, b"hello")]
    finally:
        client.close(); server.close()


def test_server_discards_wrong_manifest_without_callback():
    seen = []
    server = PersistentRaftServer("127.0.0.1", 0, manifest_hash=HASH,
                                  on_frame=lambda *args: seen.append(args)).start()
    client = PersistentRaftClient({1: RaftPeer(*server.address)}, manifest_hash="b" * 64)
    try:
        client.send(1, b"wrong")
        time.sleep(.05)
        assert seen == []
    finally:
        client.close(); server.close()
