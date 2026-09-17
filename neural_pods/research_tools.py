"""Typed tool contract for Research Pods.

Tool calls are data, not free-form model instructions. Every call carries the
namespace, Pod identity and optional generation so the lifecycle barrier can
reject stale or unauthorized research actions before execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ResearchTool:
    name: str
    purpose: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    def validate(self, arguments: Mapping[str, Any]) -> None:
        missing = [key for key in self.required if key not in arguments]
        if missing:
            raise ValueError(f"{self.name} missing required fields: {missing}")
        unknown = set(arguments) - set(self.required) - set(self.optional)
        if unknown:
            raise ValueError(f"{self.name} unknown fields: {sorted(unknown)}")


RESEARCH_TOOLS: dict[str, ResearchTool] = {
    "ann_search": ResearchTool("ann_search", "semantic nearest-neighbor retrieval", ("namespace", "query"), ("pod_identity", "generation", "top_k", "filters")),
    "bm25_search": ResearchTool("bm25_search", "exact lexical retrieval", ("namespace", "query"), ("pod_identity", "generation", "top_k", "filters")),
    "metadata_filter": ResearchTool("metadata_filter", "filter Pods by typed metadata", ("namespace", "filters"), ("branch", "top_k")),
    "graph_neighbors": ResearchTool("graph_neighbors", "follow explicit Pod relations", ("knowledge_key",), ("relation", "depth", "generation")),
    "registry_snapshot": ResearchTool("registry_snapshot", "resolve and verify current lineage", ("keys",), ("principal", "generation")),
    "calculator": ResearchTool("calculator", "evaluate a bounded numeric expression", ("expression",), ("unit",)),
    "workspace_search": ResearchTool("workspace_search", "streaming literal/regex search over approved files", ("path", "pattern"), ("file_types", "context_lines", "max_results", "case_sensitive", "regex")),
}


@dataclass(frozen=True)
class ResearchToolCall:
    tool: str
    arguments: Mapping[str, Any]
    namespace: str | None = None
    pod_identity: str | None = None
    generation: str | None = None
    principal: str = "local"
    trace_id: str | None = None

    def validate(self) -> None:
        if self.tool not in RESEARCH_TOOLS:
            raise ValueError(f"unknown research tool: {self.tool}")
        RESEARCH_TOOLS[self.tool].validate(self.arguments)
        if self.tool in {"ann_search", "bm25_search", "metadata_filter"} and not self.namespace:
            raise ValueError(f"{self.tool} requires namespace context")
        if self.generation is not None and not self.pod_identity:
            raise ValueError("generation requires pod_identity")


def research_tool_manifest() -> list[dict[str, Any]]:
    """Stable manifest suitable for tool-choice SFT and runtime exposure."""
    return [{"name": t.name, "purpose": t.purpose, "required": list(t.required), "optional": list(t.optional)}
            for t in RESEARCH_TOOLS.values()]
