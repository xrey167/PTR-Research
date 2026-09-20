"""Envelope validation, without a broker.

Before signing existed, `_on_message` validated exactly one thing: that the
protocol version matched. `manifest_hash` and `principal` rode along as
metadata that anyone with broker access could set to anything, which made
the "lineage envelope" a label rather than a claim.
"""
import pytest

from neural_pods.mesh import MeshEndpoint


class Endpoint(MeshEndpoint):
    """The envelope logic on its own: no MQTT connection, no broker."""

    def __init__(self, secret=None):
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.unsigned_rejected = 0


@pytest.fixture()
def signed():
    return Endpoint("household-key")


def _envelope(**overrides):
    envelope = {"body": {"case": "c1"}, "manifest_hash": "hash-a",
                "principal": "tenant-a", "protocol_version": 1, "ts_ms": 1}
    envelope.update(overrides)
    return envelope


def test_unsigned_endpoint_accepts_anything_and_says_so():
    plain = Endpoint()
    assert plain._signature_ok(_envelope()) is True
    assert plain.secret is None


def test_signature_round_trip(signed):
    envelope = _envelope()
    signed._sign(envelope)
    assert "sig" in envelope
    assert signed._signature_ok(envelope) is True


def test_missing_signature_is_refused(signed):
    assert signed._signature_ok(_envelope()) is False


def test_tampering_with_the_principal_breaks_the_signature(signed):
    envelope = _envelope()
    signed._sign(envelope)
    envelope["principal"] = "tenant-b"       # the field the ACL trusts
    assert signed._signature_ok(envelope) is False


def test_tampering_with_the_body_breaks_the_signature(signed):
    envelope = _envelope()
    signed._sign(envelope)
    envelope["body"]["case"] = "c2"
    assert signed._signature_ok(envelope) is False


def test_a_different_key_does_not_verify(signed):
    envelope = _envelope()
    Endpoint("other-household")._sign(envelope)
    assert signed._signature_ok(envelope) is False


def test_canonical_form_ignores_key_order_and_excludes_the_signature(signed):
    a = {"principal": "p", "body": 1, "manifest_hash": "h", "protocol_version": 1}
    b = {"protocol_version": 1, "manifest_hash": "h", "body": 1, "principal": "p"}
    assert MeshEndpoint._canonical(a) == MeshEndpoint._canonical(b)
    signed._sign(a)
    assert b"sig" not in MeshEndpoint._canonical(a)
