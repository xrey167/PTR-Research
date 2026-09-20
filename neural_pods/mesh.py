"""Pod mesh: MQTT-backed presence, discovery and messaging between pods.

Each pod announces itself under ``np/presence/{pod_id}`` (retained) and
exchanges messages on ``np/{pod_id}/{channel}``. Every payload travels in
an envelope carrying the pod's manifest hash, the protocol version and the
principal — the same lineage/ACL contract as PodRequest, transported over
MQTT. Subscriptions are matched by paho's topic filters (``+``/``#``).

Fail-closed rules: a malformed envelope is counted and dropped (never
executed); topic access is checked against the endpoint's ACL before
publish/subscribe.

What that does and does not buy you:

  * The topic ACL is enforced in THIS client, before it publishes or
    subscribes. The broker enforces nothing, and publish_raw() bypasses the
    check by design. It is a guard rail for cooperating pods, not a security
    boundary against one that is not cooperating.
  * Without `secret`, envelope validation is a protocol-version comparison.
    `manifest_hash` and `principal` ride along as metadata and anyone who can
    reach the broker can set them to anything.
  * With `secret`, every envelope carries an HMAC-SHA256 over its canonical
    form and an unsigned or wrongly signed envelope is counted and dropped.
    That is what makes `principal` mean something. The signature model
    follows A2A v1.0's signed Agent Cards; the key is shared per household,
    not per pod, so it authenticates the household, not the individual pod.

`secret` is optional so an existing deployment keeps working, but an endpoint
that has one refuses everything unsigned — mixing signed and unsigned pods on
one topic does not silently degrade to unsigned.

HOP BUDGET AND DEADLINE TRAVEL WITH THE MESSAGE. `PodLink` has carried
`hop_budget` and `deadline_ms` since the contract was written, and
`PodLink.validate()` enforces both — but only for a call that goes through a
PodLink object. Nothing of either reached the wire, so the moment a request
crossed the mesh the budget it was supposed to be bounded by no longer
existed: the receiving pod built a fresh link, with a fresh budget, and a
loop of pods could pass work around forever while every individual hop
validated. An envelope now carries `trace_id`, `hop_budget`, `deadline_ms`,
`visited` and its ORIGINAL `ts_ms`, `forward()` is the only way to pass one
on, and the receiving side drops an envelope whose budget is spent or whose
deadline has passed (`hop_exhausted`, `expired_envelopes`).

The original `ts_ms` is what makes the deadline mean anything: re-stamping it
on each hop would give every hop the full deadline again, which is the same
defect one level down.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import threading
import time
from typing import Any, Callable

PROTOCOL_VERSION = 1


def envelope_refusal(envelope: dict, *, pod_id: str, now_ms: float | None = None,
                     forwarding: bool = False) -> str | None:
    """Why this envelope must not be acted on, or None.

    Pure and module level on purpose: the decision that bounds a message is
    the one part of the mesh that has to be testable without a broker, and a
    rule that can only be exercised against live MQTT is a rule nobody
    exercises. `_on_message` and `forward` both defer to it, so the arriving
    and the departing side cannot drift apart.

    An envelope with no `hop_budget` and no `deadline_ms` is unbounded and
    accepted on arrival — that is every message the system sent before the
    bounds existed. Forwarding one is refused, because passing an unbounded
    message further is precisely what creates the loop.

    "malformed" is a RESULT, not an exception. These three fields come off
    the wire, so a sender controls their types, and this function does
    int()/float() on them. Raising here reached the paho callback, which
    re-raises into its network loop (suppress_exceptions defaults to False in
    2.1.0) and can end the receiving thread — a field added to BOUND a
    request would have become a way to silence the pod receiving it. Naming
    the case means both the arriving and the departing side refuse it the
    same way and count it, and it can be tested without a broker.
    """
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    budget = envelope.get("hop_budget")
    visited = envelope.get("visited") or ()
    deadline = envelope.get("deadline_ms")
    started = envelope.get("ts_ms")

    if not isinstance(visited, (list, tuple)):
        return "malformed"
    if deadline is not None and started is not None:
        try:
            expired = now_ms > float(started) + float(deadline)
        except (TypeError, ValueError):
            return "malformed"
        if expired:
            return "expired"
    elif deadline is not None or started is not None:
        # One half of the pair alone cannot be checked. That is not malformed
        # on arrival (ts_ms is stamped by _publish_raw, so a deadline without
        # it is an old sender), it just leaves the deadline unenforced.
        pass
    if budget is None:
        return "unbounded" if forwarding else None
    try:
        remaining = int(budget) - 1 if forwarding else int(budget)
    except (TypeError, ValueError):
        return "malformed"
    if forwarding and pod_id in tuple(visited):
        return "cycle"
    if remaining <= 0:
        return "hop_exhausted"
    return None
PRESENCE_PREFIX = "np/presence/"


class MeshACLError(PermissionError):
    """A publish/subscribe tried to leave the endpoint's allowed topics."""


class MeshEndpoint:
    """One pod's MQTT mesh connection: presence, publish, subscribe."""

    def __init__(self, broker_host: str, pod_id: str, *, principal: str = "local",
                 manifest_hash: str = "", allowed_topics: tuple[str, ...] | None = None,
                 port: int = 1883, heartbeat_s: float = 5.0,
                 secret: bytes | str | None = None):
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
        self.secret = (secret.encode("utf-8") if isinstance(secret, str) else secret)
        self.bad_envelopes = 0
        self.unsigned_rejected = 0
        self.published = 0
        self.received = 0
        self.hop_exhausted = 0
        self.expired_envelopes = 0
        self.forwarded = 0
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
    def publish(self, channel: str, body: Any, *, target_pod: str | None = None,
                link: Any = None, hop_budget: int | None = None,
                deadline_ms: int | None = None,
                trace_id: str | None = None) -> None:
        """Publish one message, optionally under a link contract's bounds.

        With `link` (a PodLink) the envelope carries that contract's trace id,
        hop budget and deadline onto the wire, so the next pod inherits the
        bounds instead of starting fresh ones. The three can also be given
        directly for a caller that has no PodLink at hand.
        """
        topic = f"np/{target_pod}/{channel}" if target_pod else f"np/{self.pod_id}/{channel}"
        if not self._topic_allowed(topic):
            raise MeshACLError(f"topic not allowed for {self.pod_id}: {topic}")
        envelope: dict[str, Any] = {"body": body}
        if link is not None:
            hop_budget = hop_budget if hop_budget is not None else getattr(link, "hop_budget", None)
            deadline_ms = deadline_ms if deadline_ms is not None else getattr(link, "deadline_ms", None)
            trace_id = trace_id or getattr(link, "trace_id", None)
            envelope["visited"] = list(getattr(link, "visited", ()) or ())
        if hop_budget is not None:
            envelope["hop_budget"] = int(hop_budget)
            envelope.setdefault("visited", [])
        if deadline_ms is not None:
            envelope["deadline_ms"] = int(deadline_ms)
        if trace_id is not None:
            envelope["trace_id"] = trace_id
        self._publish_raw(topic, envelope)

    def forward(self, topic: str, envelope: dict[str, Any], *,
                body: Any = None) -> bool:
        """Pass a received envelope one hop further. False when refused.

        The budget is spent HERE, and the original `ts_ms` is kept so the
        deadline keeps counting from when the work was requested rather than
        from the last hop. A pod that already appears in `visited` refuses:
        that is a cycle, and a cycle with a budget is only a slower loop.
        """
        visited = list(envelope.get("visited") or ())
        refusal = envelope_refusal(envelope, pod_id=self.pod_id, forwarding=True)
        if refusal == "unbounded":
            raise ValueError(
                "cannot forward an envelope without a hop budget: forwarding "
                "an unbounded message is what this method exists to prevent")
        if refusal == "malformed":
            with self._lock:
                self.bad_envelopes += 1
            return False
        if refusal == "expired":
            with self._lock:
                self.expired_envelopes += 1
            return False
        if refusal is not None:
            with self._lock:
                self.hop_exhausted += 1
            return False
        budget = envelope["hop_budget"]
        if not self._topic_allowed(topic):
            raise MeshACLError(f"topic not allowed for {self.pod_id}: {topic}")
        onward = {key: value for key, value in envelope.items()
                  if key not in ("sig",)}
        onward["hop_budget"] = int(budget) - 1
        onward["visited"] = [*visited, self.pod_id]
        if body is not None:
            onward["body"] = body
        # Everything else is re-stamped by _publish_raw EXCEPT ts_ms, which is
        # already present and must stay at its original value.
        onward["principal"] = self.principal
        onward["manifest_hash"] = self.manifest_hash
        self._publish_raw(topic, onward)
        with self._lock:
            self.forwarded += 1
        return True

    def publish_raw(self, topic: str, body: Any, *, retain: bool = False) -> None:
        """Publish on an arbitrary np/ topic, BYPASSING this endpoint's topic
        ACL; the caller (e.g. the native comm executor's egress ACL) is
        responsible for access control. The envelope is still signed when the
        endpoint has a key."""
        self._publish_raw(topic, {"body": body}, retain=retain)

    @staticmethod
    def _canonical(envelope: dict) -> bytes:
        """Signed bytes: the whole envelope except the signature itself."""
        return json.dumps({k: v for k, v in envelope.items() if k != "sig"},
                          sort_keys=True, separators=(",", ":"),
                          default=str).encode("utf-8")

    def _sign(self, envelope: dict) -> None:
        if self.secret is not None:
            envelope["sig"] = hmac.new(self.secret, self._canonical(envelope),
                                       hashlib.sha256).hexdigest()

    def _signature_ok(self, envelope: dict) -> bool:
        """An endpoint with a key accepts nothing without a matching one."""
        if self.secret is None:
            return True
        signature = envelope.get("sig")
        if not isinstance(signature, str):
            return False
        return hmac.compare_digest(
            signature, hmac.new(self.secret, self._canonical(envelope),
                                hashlib.sha256).hexdigest())

    def _publish_raw(self, topic: str, envelope: dict, *, retain: bool = False) -> None:
        envelope.setdefault("manifest_hash", self.manifest_hash)
        envelope.setdefault("principal", self.principal)
        envelope.setdefault("protocol_version", self.protocol_version)
        envelope.setdefault("ts_ms", int(time.time() * 1000))
        self._sign(envelope)
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
            # its own artifact lineage), so a foreign hash is not rejected
            # here - but it must be PRESENT, and with a key it must also be
            # covered by the signature, which is what stops anyone with
            # broker access from inventing a principal.
            for field in ("manifest_hash", "principal"):
                if field not in envelope:
                    raise ValueError(f"envelope without {field}")
            if not self._signature_ok(envelope):
                raise PermissionError("envelope signature missing or invalid")
            # Bounds carried by the envelope are enforced on ARRIVAL, before
            # any handler sees the message. A handler that answers is one more
            # hop, so letting an out-of-budget or expired message through
            # would make both bounds advisory.
            #
            # INSIDE the guard, and that is the whole point: hop_budget,
            # deadline_ms and visited come off the wire, and envelope_refusal
            # does int()/float() on them. A sender putting `hop_budget: "x"`
            # in an envelope raised ValueError out of this callback — and paho
            # 2.1.0 re-raises callback exceptions into its network loop
            # (suppress_exceptions defaults to False), which can end the
            # loop_start() thread and stop the pod receiving anything at all.
            # The malformed message also bypassed bad_envelopes, so nothing
            # counted it. A field added to bound a request must not become a
            # way to silence the pod that receives it.
            refusal = envelope_refusal(envelope, pod_id=self.pod_id)
        except PermissionError:
            with self._lock:
                self.bad_envelopes += 1
                self.unsigned_rejected += 1
            return
        except Exception:
            with self._lock:
                self.bad_envelopes += 1
            return
        if refusal == "malformed":
            # Counted with the other envelopes this endpoint could not read,
            # rather than as a spent hop budget: nothing about the budget is
            # known when its field does not parse.
            with self._lock:
                self.bad_envelopes += 1
            return
        if refusal == "expired":
            with self._lock:
                self.expired_envelopes += 1
            return
        if refusal is not None:
            with self._lock:
                self.hop_exhausted += 1
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
                    # Reported, not implied: an unsigned endpoint authenticates
                    # nothing, and the topic ACL is this client's own rule.
                    "signed": self.secret is not None,
                    "unsigned_rejected": self.unsigned_rejected,
                    "topic_acl_enforced_by": "client",
                    "forwarded": self.forwarded,
                    "hop_exhausted": self.hop_exhausted,
                    "expired_envelopes": self.expired_envelopes,
                    "subscriptions": len(self._handlers)}
