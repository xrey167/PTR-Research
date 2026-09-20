import os
import socket
import threading
import time

import pytest

from neural_pods.mesh import MeshACLError, MeshEndpoint, PROTOCOL_VERSION

BROKER_HOST = os.environ.get("NEURAL_PODS_BROKER", "10.50.0.121")
BROKER_PORT = int(os.environ.get("NEURAL_PODS_BROKER_PORT", "1883"))


def _broker_reachable(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


# These tests need a live Mosquitto. Without one every MeshEndpoint call ends
# in a bare TimeoutError, which reads like a mesh regression; skip instead,
# the same way test_raft_cluster_binding skips without the Rust wheel.
pytestmark = pytest.mark.skipif(
    not _broker_reachable(BROKER_HOST, BROKER_PORT),
    reason=f"no MQTT broker at {BROKER_HOST}:{BROKER_PORT} "
           "(set NEURAL_PODS_BROKER / NEURAL_PODS_BROKER_PORT to point at one)")


@pytest.fixture(scope="module")
def broker_host():
    return BROKER_HOST


def test_presence_discovery_between_endpoints(broker_host):
    a = MeshEndpoint(broker_host, "mesh-a", manifest_hash="hash-a")
    b = MeshEndpoint(broker_host, "mesh-b", manifest_hash="hash-b")
    try:
        peer = a.wait_for_peer("mesh-b", timeout_s=10.0)
        assert peer is not None
        assert peer["state"] == "online"
        assert peer["manifest_hash"] == "hash-b"
        assert peer["protocol_version"] == PROTOCOL_VERSION
    finally:
        a.close()
        b.close()


def test_publish_subscribe_roundtrip(broker_host):
    a = MeshEndpoint(broker_host, "mesh-a")
    received = []
    done = threading.Event()

    def on_msg(topic, envelope):
        received.append((topic, envelope))
        done.set()

    b = MeshEndpoint(broker_host, "mesh-b")
    try:
        b.subscribe("np/mesh-b/#", on_msg)
        time.sleep(0.3)  # subscription propagation
        a.publish("announce", {"value": 42}, target_pod="mesh-b")
        assert done.wait(10.0)
        topic, envelope = received[0]
        assert topic == "np/mesh-b/announce"
        assert envelope["body"]["value"] == 42
        assert envelope["principal"] == "local"
        assert envelope["protocol_version"] == PROTOCOL_VERSION
    finally:
        a.close()
        b.close()


def test_topic_acl_enforced(broker_host):
    a = MeshEndpoint(broker_host, "mesh-a", allowed_topics=("np/mesh-a/#",))
    try:
        with pytest.raises(MeshACLError):
            a.publish("intrude", {}, target_pod="other-pod")
        with pytest.raises(MeshACLError):
            a.subscribe("np/forbidden/#", lambda t, e: None)
    finally:
        a.close()


def test_bad_envelope_counted_not_executed(broker_host):
    received = []
    b = MeshEndpoint(broker_host, "mesh-b")
    a = MeshEndpoint(broker_host, "mesh-a")
    try:
        b.subscribe("np/broadcast/#", lambda t, e: received.append(e))
        time.sleep(0.3)
        # raw publish of a malformed envelope on a topic b listens to
        a._publish_raw("np/broadcast/whatever", {"unexpected": "shape"})
        # make the envelope malformed by corrupting protocol version
        a._client.publish("np/broadcast/bad", '{"protocol_version": 99}')
        time.sleep(1.0)
        assert len(received) == 1  # only the valid raw envelope passed
        assert b.stats()["bad_envelopes"] >= 1  # the bad one was dropped
    finally:
        a.close()
        b.close()
