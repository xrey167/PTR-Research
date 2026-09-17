"""Routing harness for heterogeneous Model-Pods.

The harness is deliberately small and deterministic: it selects from already
registered model candidates, records outcomes, and produces capability-level
feedback. It never turns an unverified trace into training data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Mapping

@dataclass(frozen=True)
class TaskDemand:
    task_id: str
    capabilities: tuple[str, ...]
    domain: str = "general"
    max_latency_ms: float | None = None

@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    model_family: str
    capabilities: tuple[str, ...]
    cost_weight: float = 1.0
    latency_ms: float = 0.0
    artifact_key: str | None = None

@dataclass(frozen=True)
class HarnessTrace:
    task_id: str
    model_id: str
    selected_score: float
    tool_calls: tuple[str, ...]
    outcome: str
    verified: bool
    demand: TaskDemand

class RoutingHarness:
    """Select, record, and summarize model-pool executions."""
    def __init__(self, candidates: Iterable[ModelCandidate]):
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise ValueError("routing harness needs at least one model candidate")
        if len({c.model_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("model candidate ids must be unique")
        self.traces: list[HarnessTrace] = []

    @staticmethod
    def score(demand: TaskDemand, candidate: ModelCandidate) -> float:
        wanted, offered = set(demand.capabilities), set(candidate.capabilities)
        if not wanted:
            return 0.0
        overlap = len(wanted & offered) / len(wanted)
        latency_penalty = 0.0
        if demand.max_latency_ms is not None and candidate.latency_ms > demand.max_latency_ms:
            latency_penalty = min(0.5, (candidate.latency_ms - demand.max_latency_ms) / max(demand.max_latency_ms, 1.0))
        return overlap - latency_penalty - 0.01 * candidate.cost_weight

    def select(self, demand: TaskDemand) -> ModelCandidate:
        if not demand.capabilities:
            raise ValueError("task demand needs at least one capability")
        return max(self.candidates, key=lambda c: (self.score(demand, c), c.model_id))

    def record(self, demand: TaskDemand, candidate: ModelCandidate, *, tool_calls: Iterable[str],
               outcome: str, verified: bool) -> HarnessTrace:
        if candidate not in self.candidates:
            raise ValueError("candidate is not registered in this harness")
        trace = HarnessTrace(demand.task_id, candidate.model_id, self.score(demand, candidate),
                             tuple(tool_calls), outcome, bool(verified), demand)
        self.traces.append(trace)
        return trace

    def capability_feedback(self) -> Mapping[str, Mapping[str, float]]:
        """Aggregate only verified traces for the next training mixture."""
        stats: dict[str, dict[str, float]] = {}
        for trace in self.traces:
            if not trace.verified:
                continue
            for capability in trace.demand.capabilities:
                row = stats.setdefault(capability, {"verified": 0.0, "success": 0.0})
                row["verified"] += 1.0
                if trace.outcome.lower() in {"success", "passed", "correct"}:
                    row["success"] += 1.0
        for row in stats.values():
            row["success_rate"] = row["success"] / row["verified"] if row["verified"] else 0.0
        return stats
