"""Iterative local search episodes, without a hosted search API.

The agent is intentionally policy-agnostic. A learned policy can emit
``SearchAction`` objects later; the included policy is a deterministic baseline
that makes the complete tool loop measurable now. Each turn fans out calls to
the local backend and observes only returned candidates, just like a search
tool used by a model.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import re
import time
from typing import Any, Callable, Iterable, Mapping

from .local_search import LocalSearchBackend, SearchHit


@dataclass(frozen=True)
class SearchAction:
    tool: str  # bm25, ann, hybrid, regex
    text: str = ""
    vector: tuple[float, ...] | None = None
    regex: str | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)
    top_k: int = 10


@dataclass
class SearchEpisode:
    question: str
    target_keys: tuple[str, ...]
    turns: list[list[SearchAction]] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)
    found_keys: tuple[str, ...] = ()
    recall: float = 0.0
    mrr: float = 0.0
    reward: float = 0.0
    latency_score: float = 0.0
    elapsed_s: float = 0.0
    search_calls: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "turn_limit"


class LocalSearchAgent:
    """Run bounded, parallel, iterative retrieval against a local backend."""

    def __init__(self, backend: LocalSearchBackend, namespace: str = "default", branch: str = "main",
                 *, max_turns: int = 4, parallelism: int = 8, latency_budget_s: float = 5.0):
        self.backend, self.namespace, self.branch = backend, namespace, branch
        if max_turns < 1 or parallelism < 1 or latency_budget_s <= 0:
            raise ValueError("max_turns, parallelism and latency_budget_s must be positive")
        self.max_turns, self.parallelism, self.latency_budget_s = max_turns, parallelism, latency_budget_s

    @staticmethod
    def baseline_policy(question: str, turn: int, observed: list[SearchHit]) -> list[SearchAction]:
        """A transparent policy baseline: broad, exact and entity-focused calls."""
        words = re.findall(r"[\w-]+", question, re.UNICODE)
        actions = [SearchAction("bm25", text=question, top_k=10)]
        if turn == 0 and len(words) > 3:
            actions.append(SearchAction("bm25", text=" ".join(words[: max(2, len(words) // 2)]), top_k=10))
        if turn > 0 and observed:
            # Search the most informative returned title/text terms again. This
            # is a deterministic stand-in for model query reformulation.
            seed = " ".join(re.findall(r"[\w-]+", observed[0].text)[:8])
            if seed: actions.append(SearchAction("bm25", text=seed, top_k=10))
        return actions

    @staticmethod
    def adaptive_policy(question: str, turn: int, observed: list[SearchHit]) -> list[SearchAction]:
        """SID-style deterministic teacher policy for RL rollouts.

        It deliberately fans out narrow and broad lexical searches, then uses
        terms discovered in the previous turn. A learned policy can replace
        this function while consuming the same SearchAction/observation trace.
        """
        words = re.findall(r"[\w-]+", question, re.UNICODE)
        actions = [SearchAction("bm25", text=question, top_k=20)]
        if len(words) > 4:
            actions.append(SearchAction("bm25", text=" ".join(words[:4]), top_k=20))
            actions.append(SearchAction("bm25", text=" ".join(words[-4:]), top_k=20))
        if turn and observed:
            for hit in observed[:2]:
                seed = " ".join(re.findall(r"[\w-]+", hit.text)[:6])
                if seed:
                    actions.append(SearchAction("bm25", text=seed, top_k=20))
        return actions

    def _run_action(self, action: SearchAction) -> list[SearchHit]:
        if action.tool == "bm25":
            return self.backend.search(self.namespace, branch=self.branch, text=action.text,
                                       filters=action.filters, top_k=action.top_k,
                                       vector_weight=0.0, lexical_weight=1.0)
        if action.tool == "ann":
            return self.backend.search(self.namespace, branch=self.branch, vector=action.vector,
                                       filters=action.filters, top_k=action.top_k,
                                       vector_weight=1.0, lexical_weight=0.0)
        if action.tool == "hybrid":
            return self.backend.search(self.namespace, branch=self.branch, text=action.text,
                                       vector=action.vector, filters=action.filters, top_k=action.top_k)
        if action.tool == "regex":
            return self.backend.search(self.namespace, branch=self.branch, regex=action.regex,
                                       filters=action.filters, top_k=action.top_k,
                                       vector_weight=0.0, lexical_weight=0.0)
        raise ValueError(f"unknown search tool: {action.tool}")

    def run(self, question: str, target_keys: Iterable[str],
            policy: Callable[[str, int, list[SearchHit]], list[SearchAction]] | None = None) -> SearchEpisode:
        targets = tuple(dict.fromkeys(target_keys))
        if not targets:
            raise ValueError("an episode needs at least one target key")
        policy = policy or self.baseline_policy
        started = time.perf_counter()
        episode = SearchEpisode(question, targets)
        merged: dict[str, SearchHit] = {}
        observed: list[SearchHit] = []
        for turn in range(self.max_turns):
            actions = list(policy(question, turn, observed))
            if not actions:
                break
            episode.turns.append(actions)
            for action in actions:
                episode.tool_counts[action.tool] = episode.tool_counts.get(action.tool, 0) + 1
            with ThreadPoolExecutor(max_workers=min(self.parallelism, len(actions))) as pool:
                batches = list(pool.map(self._run_action, actions))
            episode.search_calls += len(actions)
            for batch in batches:
                for hit in batch:
                    previous = merged.get(hit.key)
                    if previous is None or hit.score > previous.score:
                        merged[hit.key] = hit
            observed = sorted(merged.values(), key=lambda hit: (-hit.score, hit.key))
            if set(targets) <= set(merged):
                episode.stop_reason = "targets_found"
                break
        ranked = sorted(merged.values(), key=lambda hit: (-hit.score, hit.key))
        target_set = set(targets)
        found = tuple(hit.key for hit in ranked if hit.key in target_set)
        episode.hits = ranked
        episode.found_keys = found
        episode.recall = len(found) / len(targets)
        reciprocal = next((1.0 / (i + 1) for i, hit in enumerate(ranked) if hit.key in target_set), 0.0)
        episode.mrr = reciprocal
        # Recall is primary; reciprocal rank rewards useful ordering; the small
        # latency term encourages a policy to stop as soon as it has enough.
        episode.elapsed_s = time.perf_counter() - started
        episode.latency_score = max(0.0, 1.0 - episode.elapsed_s / self.latency_budget_s)
        episode.reward = episode.recall * 0.7 + episode.mrr * 0.2 + episode.latency_score * 0.1
        return episode
