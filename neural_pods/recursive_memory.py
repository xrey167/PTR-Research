"""Replayable experience Pods for recursive exploration.

This is a local, deterministic substrate inspired by the two linked ideas:
historical discovery trees become an offline replay simulator, while new
exploration proceeds broad first and deepens only uncertain or failed paths.
It stores action-condition-outcome records, not hidden model claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Experience:
    experience_id: str
    environment: str
    condition: str
    action: str
    outcome: str
    verified: bool
    depth: int = 0
    parent_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperiencePod:
    """A compact, reusable memory Pod produced from verified experience."""

    pod_id: str
    environment: str
    condition: str
    action: str
    outcome: str
    confidence: float
    source_experience_ids: tuple[str, ...]
    depth: int
    semantic_type: str = "context"

    def to_payload(self) -> dict:
        return {
            "content": f"If {self.condition}, perform {self.action}: {self.outcome}",
            "semantic_type": self.semantic_type,
            "tags": ["experience", "verified", self.environment],
            "experience": {
                "condition": self.condition,
                "action": self.action,
                "outcome": self.outcome,
                "confidence": self.confidence,
                "source_experience_ids": list(self.source_experience_ids),
                "depth": self.depth,
            },
        }


@dataclass(frozen=True)
class ReplayScore:
    evaluated: int
    correct: int
    coverage: float
    accuracy: float


class DiscoveryReplay:
    """Offline replay simulator over recorded discovery experiences."""

    def __init__(self, experiences: Iterable[Experience]):
        self.experiences = tuple(experiences)
        self._verified = tuple(x for x in self.experiences if x.verified)

    def score(self, policy: Callable[[str, str], str], *, environment: str | None = None) -> ReplayScore:
        rows = [x for x in self._verified if environment is None or x.environment == environment]
        if not rows:
            return ReplayScore(0, 0, 0.0, 0.0)
        correct = sum(policy(x.condition, x.action) == x.outcome for x in rows)
        covered = sum(bool(policy(x.condition, x.action)) for x in rows)
        return ReplayScore(len(rows), correct, covered / len(rows), correct / len(rows))

    def discovery_tree(self, root_id: str) -> tuple[Experience, ...]:
        by_parent: dict[str | None, list[Experience]] = {}
        for row in self.experiences:
            by_parent.setdefault(row.parent_id, []).append(row)
        found: list[Experience] = []
        frontier = [root_id]
        while frontier:
            current = frontier.pop(0)
            row = next((x for x in self.experiences if x.experience_id == current), None)
            if row is None:
                continue
            found.append(row)
            frontier.extend(x.experience_id for x in by_parent.get(current, ()))
        return tuple(found)


class BroadThenDeepExplorer:
    """Collect broad candidates, then expand only weak branches."""

    def __init__(self, *, max_depth: int = 3, confidence_threshold: float = 0.75):
        if max_depth < 1 or not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("invalid exploration limits")
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold

    def explore(self, seeds: Sequence[Experience], expand: Callable[[Experience], Sequence[Experience]]) -> tuple[Experience, ...]:
        broad = list(seeds)
        seen = {x.experience_id for x in broad}
        frontier = [x for x in broad if not x.verified or self._confidence(x) < self.confidence_threshold]
        while frontier:
            parent = frontier.pop(0)
            if parent.depth >= self.max_depth:
                continue
            for child in expand(parent):
                if child.experience_id in seen:
                    continue
                seen.add(child.experience_id)
                broad.append(child)
                if not child.verified or self._confidence(child) < self.confidence_threshold:
                    frontier.append(child)
        return tuple(broad)

    @staticmethod
    def _confidence(item: Experience) -> float:
        value = item.metadata.get("confidence", "1" if item.verified else "0")
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def consolidate(self, experiences: Iterable[Experience]) -> tuple[ExperiencePod, ...]:
        groups: dict[tuple[str, str, str, str], list[Experience]] = {}
        for item in experiences:
            if item.verified:
                groups.setdefault((item.environment, item.condition, item.action, item.outcome), []).append(item)
        pods: list[ExperiencePod] = []
        for (environment, condition, action, outcome), rows in sorted(groups.items()):
            confidence = sum(self._confidence(x) for x in rows) / len(rows)
            pod_id = f"experience:{environment}:{condition}:{action}:{outcome}"
            pods.append(ExperiencePod(pod_id, environment, condition, action, outcome,
                                      confidence, tuple(x.experience_id for x in rows),
                                      min(x.depth for x in rows)))
        return tuple(pods)
