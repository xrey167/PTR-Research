"""Broad-to-deep search episodes that emit replayable Experience records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .recursive_memory import BroadThenDeepExplorer, Experience, ExperiencePod
from .search_agent import LocalSearchAgent, SearchAction, SearchEpisode


@dataclass(frozen=True)
class RecursiveSearchResult:
    question: str
    target_keys: tuple[str, ...]
    broad: SearchEpisode
    deep: SearchEpisode | None
    experiences: tuple[Experience, ...]
    pods: tuple[ExperiencePod, ...]


class RecursiveSearchRunner:
    """Run broad discovery first, then deepen only incomplete searches."""

    def __init__(self, backend, namespace: str = "default", branch: str = "main",
                 *, broad_parallelism: int = 8, deep_parallelism: int = 8,
                 max_deep_turns: int = 4):
        self.backend = backend
        self.namespace = namespace
        self.branch = branch
        self.broad_parallelism = broad_parallelism
        self.deep_parallelism = deep_parallelism
        self.max_deep_turns = max_deep_turns

    @staticmethod
    def _broad_policy(question: str, turn: int, observed) -> list[SearchAction]:
        # One cheap wide probe is intentionally shallow; missing evidence is
        # the signal that triggers the deeper policy below.
        return [SearchAction("bm25", text=question, top_k=1)]

    @staticmethod
    def _deep_policy(question: str, turn: int, observed) -> list[SearchAction]:
        actions = LocalSearchAgent.baseline_policy(question, turn, observed)
        if turn == 0:
            # Broad lexical exploration plus a deliberately narrower query.
            actions.append(SearchAction("bm25", text=question, top_k=20))
        elif observed:
            terms = " ".join(observed[0].text.split()[:6])
            if terms:
                actions.append(SearchAction("regex", regex=r"(?i)" + r"|".join(terms.split()), top_k=20))
        return actions

    @staticmethod
    def _records(episode: SearchEpisode, *, parent: str | None, verified: bool) -> list[Experience]:
        found = ",".join(episode.found_keys) or "none"
        records = []
        for index, actions in enumerate(episode.turns):
            for offset, action in enumerate(actions):
                records.append(Experience(
                    experience_id=f"{episode.question[:24]}:{index}:{offset}",
                    environment="local-search",
                    condition=episode.question,
                    action=f"{action.tool}:{action.text or action.regex or ''}",
                    outcome=f"found:{found}",
                    verified=verified,
                    depth=index,
                    parent_id=parent,
                    metadata={"confidence": str(episode.recall)},
                ))
        return records

    def run(self, question: str, target_keys: Iterable[str]) -> RecursiveSearchResult:
        targets = tuple(dict.fromkeys(target_keys))
        if not targets:
            raise ValueError("a recursive search task needs target keys")
        broad_agent = LocalSearchAgent(self.backend, self.namespace, self.branch,
                                       max_turns=1, parallelism=self.broad_parallelism)
        broad = broad_agent.run(question, targets, policy=self._broad_policy)
        experiences = self._records(broad, parent=None, verified=broad.recall == 1.0)
        deep = None
        if broad.recall < 1.0:
            deep_agent = LocalSearchAgent(self.backend, self.namespace, self.branch,
                                          max_turns=self.max_deep_turns,
                                          parallelism=self.deep_parallelism)
            deep = deep_agent.run(question, targets, policy=self._deep_policy)
            parent = experiences[0].experience_id if experiences else None
            experiences.extend(self._records(deep, parent=parent, verified=deep.recall == 1.0))
        explorer = BroadThenDeepExplorer(max_depth=self.max_deep_turns)
        pods = explorer.consolidate(experiences)
        return RecursiveSearchResult(question, targets, broad, deep, tuple(experiences), pods)
