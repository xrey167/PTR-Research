"""Efficient Pod-to-Pod branch/merge and duplex session primitives.

Multiplexing happens at the typed Pod-result level, never by blindly sharing
hidden states between different model families. Duplex is an event protocol
usable over SSH, WebSocket or an in-process adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .registry import InvalidState


@dataclass(frozen=True)
class HypothesisBranch:
    branch_id: str
    pod_id: str
    generation: str
    evidence: tuple[str, ...]
    answer: str | None
    confidence: float
    latency_ms: float
    verified: bool = False

    def __post_init__(self):
        if not self.branch_id or not self.pod_id or not self.generation:
            raise ValueError("branch identity is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.latency_ms < 0:
            raise ValueError("latency cannot be negative")


@dataclass(frozen=True)
class MergedPodResult:
    answer: str | None
    evidence: tuple[str, ...]
    source_branches: tuple[str, ...]
    confidence: float
    merge_policy: str


def merge_branches(branches: Iterable[HypothesisBranch], *, max_evidence: int = 20,
                   require_verified: bool = True) -> MergedPodResult:
    """Confidence-weighted typed branch merge with deterministic tie-breaking."""
    items = tuple(branches)
    if not items:
        raise InvalidState("cannot merge an empty Pod branch set")
    usable = tuple(b for b in items if b.verified or not require_verified)
    if not usable:
        raise InvalidState("no verified Pod branches available")
    ranked = sorted(usable, key=lambda b: (-b.confidence, b.latency_ms, b.branch_id))
    evidence = []
    for branch in ranked:
        for key in branch.evidence:
            if key not in evidence:
                evidence.append(key)
            if len(evidence) >= max_evidence:
                break
        if len(evidence) >= max_evidence:
            break
    winner = ranked[0]
    confidence = sum(b.confidence for b in ranked) / len(ranked)
    return MergedPodResult(winner.answer, tuple(evidence), tuple(b.branch_id for b in ranked),
                           confidence, "verified_confidence_then_latency")


@dataclass(frozen=True)
class PodEvent:
    event_type: str
    sequence: int
    payload: Mapping[str, object] = field(default_factory=dict)


class DuplexSession:
    """Minimal model-neutral duplex state machine."""
    def __init__(self, session_id: str, *, capabilities: Iterable[str] = ()):
        if not session_id:
            raise ValueError("session_id required")
        self.session_id = session_id
        self.capabilities = frozenset(capabilities)
        self.epoch = 0
        self.turn_id = 0
        self._sequence = 0
        self._events: list[PodEvent] = []
        self.closed = False

    def emit(self, event_type: str, **payload) -> PodEvent:
        if self.closed:
            raise InvalidState("duplex session is closed")
        self._sequence += 1
        event = PodEvent(event_type, self._sequence, {"session_id": self.session_id, "epoch": self.epoch, **payload})
        self._events.append(event)
        return event

    def commit_turn(self) -> PodEvent:
        self.turn_id += 1
        return self.emit("turn.commit", turn_id=self.turn_id)

    def interrupt(self, reason="barge_in") -> PodEvent:
        self.epoch += 1
        return self.emit("response.interrupted", reason=reason)

    def resume(self, last_sequence: int) -> tuple[PodEvent, ...]:
        if last_sequence < 0 or last_sequence > self._sequence:
            raise InvalidState("invalid resume cursor")
        return tuple(self._events[last_sequence:])

    def close(self) -> PodEvent:
        event = self.emit("session.closed")
        self.closed = True
        return event
