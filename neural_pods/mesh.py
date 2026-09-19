"""Pod mesh: MQTT-backed presence, discovery and messaging between pods.

Each pod announces itself under ``np/presence/{pod_id}`` (retained) and
exchanges messages on ``np/{pod_id}/{channel}``. Every payload travels in
an envelope carrying the pod's manifest hash, the protocol version and the
principal — the same lineage/ACL contract as PodRequest, transported over
MQTT. Subscriptions are matched by paho's topic filters (``+``/``#``).

Fail-closed rules: a malformed envelope is counted and dropped (never
executed); topic access is checked against the endpoint's ACL before
publish/subscribe.
"""
from __future__ import annotations
import json
import threading
import time
from typing import Any, Callable

PROTOCOL_VERSION = 1
PRESENCE_PREFIX = "np/presence/"


class MeshACLError(PermissionError):
    """A publish/subscribe tried to leave the endpoint's allowed topics."""


class MeshEndpoint:
    """One pod's MQTT mesh connection: presence, publish, subscribe."""

    def __init__(self, broker_host: str, pod_id: str, *, principal: str = "local",
                 manifest_hash: str = "", allowed_topics: tuple[str, ...] | None = None,
                 port: int = 1883, heartbeat_s: float = 5.0):
        import paho.mqtt.client as mqtt
        if allowed_topics is None:
            # Default: unrestricted pod mesh access. A pod's link_contract
            # narrows this (egress ACL) exactly like PodTransport does.
            allowed_topics = ("np/#",)
        self.pod_id = pod_id
        self.principal = principal
        self.manifest_hash = manifest_hash
        self.allowed_topics = tuple(allowed_topics)
        self.heartbeat_s = float(heartbeat_s)
        self.protocol_version = PROTOCOL_VERSION
        self.bad_envelopes = 0
        self.published = 0
        self.received = 0
        self._handlers: dict[str, Callable[[dict], None]] = {}
        self._lock = threading.RLock()
        # Unique client id: a reused pod_id must never collide with a zombie
        # connection from a previous run (broker would kick sessions and
        # drop QoS traffic between the flapping connections).
        import uuid
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                   client_id=f"{pod_id}-{uuid.uuid4().hex[:8]}")
        self._client.on_message = self._on_message
        self._client.connect(broker_host, port, keepalive=30)
        self._client.loop_start()
        self._presence_topic = PRESENCE_PREFIX + pod_id
        self.announce()
        self._heartbeat_stop = threading.Event()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    # --- topic ACL -------------------------------------------------------
    def _topic_allowed(self, topic: str) -> bool:
        for pattern in self.allowed_topics:
            pattern_parts = pattern.split("/")
            topic_parts = topic.split("/")
            ok = True
            for i, part in enumerate(pattern_parts):
                if part == "#":
                    ok = True
                    break
                if part == "+":
                    continue
                if i >= len(topic_parts) or topic_parts[i] != part:
                    ok = False
                    break
            else:
                ok = len(topic_parts) == len(pattern_parts)
            if ok:
                return True
        return False

    # --- presence --------------------------------------------------------
    def announce(self) -> None:
        self._publish_raw(self._presence_topic, {
            "pod_id": self.pod_id, "principal": self.principal,
            "manifest_hash": self.manifest_hash,
            "protocol_version": self.protocol_version, "state": "online",
        }, retain=True)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_s):
            try:
                self.announce()
            except Exception:
                pass

    def close(self) -> None:
        self._heartbeat_stop.set()
        try:
            self._publish_raw(self._presence_topic, {
                "pod_id": self.pod_id, "state": "offline",
                "protocol_version": self.protocol_version,
            }, retain=True)
        except Exception:
            pass
        self._client.loop_stop()
        self._client.disconnect()

    # --- messaging -------------------------------------------------------
    def publish(self, channel: str, body: Any, *, target_pod: str | None = None) -> None:
        topic = f"np/{target_pod}/{channel}" if target_pod else f"np/{self.pod_id}/{channel}"
        if not self._topic_allowed(topic):
            raise MeshACLError(f"topic not allowed for {self.pod_id}: {topic}")
        self._publish_raw(topic, {"body": body})

    def publish_raw(self, topic: str, body: Any, *, retain: bool = False) -> None:
        """Publish on an arbitrary np/ topic; the caller (e.g. the native
        comm executor's egress ACL) is responsible for access control."""
        self._publish_raw(topic, {"body": body}, retain=retain)

    def _publish_raw(self, topic: str, envelope: dict, *, retain: bool = False) -> None:
        envelope.setdefault("manifest_hash", self.manifest_hash)
        envelope.setdefault("principal", self.principal)
        envelope.setdefault("protocol_version", self.protocol_version)
        envelope.setdefault("ts_ms", int(time.time() * 1000))
        # Retry until the broker accepts (the initial announce can race the
        # connection setup and would otherwise silently vanish).
        # NEVER wait_for_publish here: callbacks (on_message handlers that
        # answer with a publish) run on the paho network loop thread, and
        # blocking that thread on its own PUBACK deadlocks the client.
        for _attempt in range(15):
            info = self._client.publish(topic, json.dumps(envelope), retain=retain, qos=1)
            if info.rc == 0:
                break
            time.sleep(0.2)
        with self._lock:
            self.published += 1

    def subscribe(self, topic_filter: str,
                  handler: Callable[[str, dict[str, Any]], None]) -> None:
        """Register a callback for a topic filter; body-only payloads arrive
        pre-validated (envelope fields verified, bad envelopes dropped)."""
        if not self._topic_allowed(topic_filter):
            raise MeshACLError(f"topic filter not allowed for {self.pod_id}: {topic_filter}")
        with self._lock:
            self._handlers[topic_filter] = handler
        self._client.subscribe(topic_filter, qos=1)

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        p_parts, t_parts = pattern.split("/"), topic.split("/")
        for i, part in enumerate(p_parts):
            if part == "#":
                return True
            if i >= len(t_parts):
                return False
            if part != "+" and part != t_parts[i]:
                return False
        return len(p_parts) == len(t_parts)

    def _on_message(self, _client, _userdata, message) -> None:
        topic = message.topic
        try:
            envelope = json.loads(message.payload.decode("utf-8"))
            if envelope.get("protocol_version") != self.protocol_version:
                raise ValueError("protocol version mismatch")
            # Manifest hashes differ legitimately between pods (each carries
            # its own artifact lineage); integrity is enforced per channel
            # and by PodTransport, not by rejecting foreign hashes here.
        except Exception:
            with self._lock:
                self.bad_envelopes += 1
            return
        with self._lock:
            self.received += 1
            handlers = [(p, h) for p, h in self._handlers.items() if self._topic_matches(p, topic)]
        for _pattern, handler in handlers:
            handler(topic, envelope)

    def wait_for_peer(self, pod_id: str, timeout_s: float = 10.0) -> dict | None:
        """Blocking discovery: wait for a peer's retained presence record."""
        found: dict | None = None
        done = threading.Event()

        def on_presence(topic: str, envelope: dict) -> None:
            nonlocal found
            if envelope.get("pod_id") == pod_id and envelope.get("state") == "online":
                found = envelope
                done.set()

        self.subscribe(f"{PRESENCE_PREFIX}{pod_id}", on_presence)
        done.wait(timeout_s)
        return found

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"pod_id": self.pod_id, "published": self.published,
                    "received": self.received, "bad_envelopes": self.bad_envelopes,
                    "protocol_version": self.protocol_version,
                    "subscriptions": len(self._handlers)}
