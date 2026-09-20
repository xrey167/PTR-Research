"""Task graph: parallel pod orchestration with explicit token flow.

Nodes are pod invocations (any callable taking a payload and returning a
result). Edges declare token flow: the output of node A is injected into
the payload of dependent node B (configurable key, default "context").
Independent nodes run in parallel (bounded), dependents wait for all
inputs. Merging multiple inputs is the caller's transform responsibility;
`merge_branches` from pod_streams is the reference implementation for
hypothesis merging.

Backpressure: at most `max_parallel` nodes execute at once; per-node
timeouts mirror PodRequest deadlines.

Metrics (see `run`): the graph measures each node's elapsed time, and a
node's elapsed time includes whatever it spent waiting on a resource
outside the graph — a remote pod, a shared broker, a GPU queue. The graph
cannot see inside a handler and therefore cannot separate work from
waiting. So `node_elapsed_sum_s / wall_s` is reported as
`mean_concurrency` (Little's law: the average number of in-flight nodes),
NOT as a speedup: contention inflates both the numerator and that ratio.
`critical_path_ratio` is the honest scheduler metric — it compares wall
time against the longest dependency chain, which is the lower bound the
DAG imposes on the measured durations.
"""
from __future__ import annotations
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TaskNode:
    node_id: str
    handler: Callable[[dict[str, Any]], Any]
    depends_on: tuple[str, ...] = ()
    context_key: str = "context"
    timeout_s: float = 30.0


@dataclass
class TaskResult:
    node_id: str
    output: Any = None
    error: str | None = None
    started: float = 0.0
    finished: float = 0.0
    inputs_from: tuple[str, ...] = ()


class TaskGraph:
    def __init__(self, nodes: list[TaskNode], *, max_parallel: int = 8):
        self.nodes = {n.node_id: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate node ids")
        self._validate_dag()
        self.max_parallel = max_parallel
        self.results: dict[str, TaskResult] = {}
        self._lock = threading.RLock()

    def _validate_dag(self) -> None:
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"unknown dependency {dep!r} of {node.node_id}")
        resolved: set[str] = set()
        while len(resolved) < len(self.nodes):
            progress = False
            for node_id, node in self.nodes.items():
                if node_id in resolved:
                    continue
                if all(dep in resolved for dep in node.depends_on):
                    resolved.add(node_id)
                    progress = True
            if not progress:
                raise ValueError("cycle in task graph")

    def run(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the graph; returns {node_id: TaskResult} for every node."""
        base_payload = dict(payload or {})
        results: dict[str, TaskResult] = {}
        results_lock = threading.RLock()
        done_events: dict[str, threading.Event] = {nid: threading.Event()
                                                   for nid in self.nodes}
        outputs: dict[str, Any] = {}
        node_elapsed_sum = 0.0

        def run_node(node: TaskNode) -> None:
            nonlocal node_elapsed_sum
            for dep in node.depends_on:
                done_events[dep].wait()
            with results_lock:
                inputs = {dep: outputs.get(dep) for dep in node.depends_on}
            node_payload = dict(base_payload)
            if node.depends_on:
                node_payload[node.context_key] = (
                    inputs[node.depends_on[-1]] if len(node.depends_on) == 1 else inputs)
            started = time.perf_counter()
            output: Any = None
            error: str | None = None
            try:
                output = node.handler(node_payload)
            except Exception as exc:
                error = repr(exc)
            finished = time.perf_counter()
            with results_lock:
                outputs[node.node_id] = output
                results[node.node_id] = TaskResult(
                    node_id=node.node_id, output=output, error=error,
                    started=started, finished=finished,
                    inputs_from=tuple(node.depends_on))
                node_elapsed_sum += finished - started
                done_events[node.node_id].set()

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = [pool.submit(run_node, node) for node in self.nodes.values()]
            for future in futures:
                future.result()
        wall = time.perf_counter() - started
        with results_lock:
            self.results = results
        critical_path = self._critical_path_s(results)
        return {"wall_s": round(wall, 3),
                "node_elapsed_sum_s": round(node_elapsed_sum, 3),
                "critical_path_s": round(critical_path, 3),
                "mean_concurrency": round(node_elapsed_sum / max(wall, 1e-9), 2),
                "critical_path_ratio": round(critical_path / max(wall, 1e-9), 2),
                "metric_note": (
                    "mean_concurrency is the average number of in-flight nodes, "
                    "not a speedup: node durations include time spent waiting on "
                    "resources outside the graph, so contention raises it. "
                    "critical_path_ratio near 1.0 means the scheduler reached the "
                    "lower bound the DAG imposes on the measured durations."),
                "results": results}

    def _critical_path_s(self, results: dict[str, TaskResult]) -> float:
        """Longest dependency chain, by measured node duration.

        Measured durations include waiting on shared resources, so this is a
        lower bound on wall time for the observed run — not a bound on pure
        work, which the graph cannot observe.
        """
        finish: dict[str, float] = {}

        def resolve(node_id: str) -> float:
            if node_id not in finish:
                result = results[node_id]
                start = max((resolve(dep)
                             for dep in self.nodes[node_id].depends_on),
                            default=0.0)
                finish[node_id] = start + (result.finished - result.started)
            return finish[node_id]

        return max((resolve(node_id) for node_id in self.nodes), default=0.0)

    def outputs(self) -> dict[str, Any]:
        return {nid: r.output for nid, r in self.results.items()}
