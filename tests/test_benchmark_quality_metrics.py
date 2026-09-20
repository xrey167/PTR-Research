"""The quality metrics that decide whether a generation may be promoted.

Three benchmarks score model answers against targets: reflex dispatch, the
gen5/gen3 ensemble router, and the heterogeneous ensemble. Between them they
produce `union_raw`, which the gate compares against a static baseline — the
number that says a new arrangement of pods is not worse than the old one.

All three needed one or two vLLM replicas on GPUs to run at all, so the
comparison of two strings could only ever be exercised on the server. What
that hid: a `union_guarded` clause reading `g3_raw and target == target`
(always true, so the intent was unreadable), a `fallback_used` counter that
could exceed `n` because a failed fallback call skipped the case after
incrementing it, and a reflex run in which not one address resolved.

`summarise()` in each takes recorded observations. No model, no GPU.
"""
import pytest

from research import benchmark_ensemble_router, benchmark_hetero_ensemble
from research import benchmark_reflex_dispatch


# ------------------------------------------------------------ reflex dispatch

def _observation(case, *, answer, target, guarded=None, pod="pod:reader-gen5",
                 alias="reader-gen5", other=None, reflex=True):
    observation = {"id": case, "raw_signal": alias, "alias": alias, "pod": pod,
                   "reflex": reflex, "answer": answer,
                   "guarded_answer": guarded if guarded is not None else answer,
                   "target": target, "latency_ms": 305.0}
    if other is not None:
        observation["other_pod"] = "pod:reader-gen6"
        observation["other_answer"] = other
    return observation


def test_reflex_scoring_counts_raw_guarded_and_union():
    observations = [
        _observation("a", answer="5 days", target="5 days"),
        _observation("b", answer="wrong", target="7 days", other="7 days"),
        _observation("c", answer="wrong", target="9 days", other="also wrong"),
        _observation("d", answer="10", target="10 days", guarded="10 days"),
    ]
    report = benchmark_reflex_dispatch.summarise(
        observations, channel_stats={"reflex_hits": 4, "failovers": 0},
        baseline_union_raw=2, elapsed_s=94.3)
    metrics = report["metrics"]
    assert metrics["n"] == 4
    assert metrics["reflex_raw"] == 1        # only "a" was exactly right
    assert metrics["reflex_guarded"] == 2    # "a" and the guarded "d"
    assert metrics["other_pod_correct"] == 1
    assert metrics["union_raw"] == 2         # "a" plus "b" via the other pod


def test_the_recorded_run_where_no_address_resolved_is_legible():
    """The 2026-09-19 evidence: 132 cases, reflex_hits 0, failovers 132.
    Every answer came from the default pod, and the old report gave no field
    that said so — the check read the quality numbers and passed."""
    observations = [_observation(f"c{i}", answer="x", target="x",
                                 pod="pod:reader-gen5", alias="unknown-alias",
                                 reflex=False) for i in range(132)]
    report = benchmark_reflex_dispatch.summarise(
        observations, channel_stats={"reflex_hits": 0, "reflex_misses": 132,
                                     "failovers": 132},
        baseline_union_raw=126, elapsed_s=94.3)
    metrics = report["metrics"]
    assert metrics["answers_from_reflex"] == 0
    assert metrics["answers_from_failover"] == 132
    assert metrics["unmapped_signals"] == 132


def test_an_empty_reflex_run_reports_no_latency():
    report = benchmark_reflex_dispatch.summarise(
        [], channel_stats={}, baseline_union_raw=126, elapsed_s=0.0)
    assert report["metrics"]["reflex_p50_ms"] is None
    assert report["metrics"]["n"] == 0


def test_only_twenty_signal_samples_are_carried():
    observations = [_observation(f"c{i}", answer="x", target="x")
                    for i in range(50)]
    report = benchmark_reflex_dispatch.summarise(
        observations, channel_stats={}, baseline_union_raw=1, elapsed_s=1.0)
    assert len(report["metrics"]["signal_samples"]) == 20


def test_reflex_dispatch_failures_remain_counted_observations():
    failed = {"id": "a", "raw_signal": "reader-gen5",
              "alias": "reader-gen5", "target": "5 days",
              "latency_ms": 1.0, "dispatch_error": True}
    other_failed = _observation("b", answer="wrong", target="7 days")
    other_failed["other_dispatch_error"] = True
    report = benchmark_reflex_dispatch.summarise(
        [failed, other_failed], channel_stats={}, baseline_union_raw=0,
        elapsed_s=1.0)
    metrics = report["metrics"]
    assert metrics["n"] == 2
    assert metrics["dispatch_errors"] == 1
    assert metrics["other_dispatch_errors"] == 1
    assert metrics["union_raw"] == 0


# ------------------------------------------------------------ ensemble router

def _case(target, primary, *, guarded=None, fallback=None):
    case = {"id": "x", "target": target, "primary_answer": primary,
            "primary_guarded": guarded if guarded is not None else primary}
    if fallback is not None:
        case["fallback_answer"] = fallback
    return case


def test_the_ensemble_union_counts_either_pod_getting_it_right():
    observations = [
        _case("5 days", "5 days"),                          # primary right
        _case("7 days", "nope", fallback="7 days"),         # fallback right
        _case("9 days", "nope", fallback="nope"),           # neither
    ]
    report = benchmark_ensemble_router.summarise(
        observations, errors=0, elapsed_s=12.0,
        router_metrics={"requests": 3}, primary="reader-gen5",
        fallback="reader-gen3")
    metrics = report["metrics"]
    assert metrics["gen5_raw"] == 1
    assert metrics["gen3_raw"] == 1
    assert metrics["union_raw"] == 2
    assert metrics["fallback_used"] == 2


def test_the_guarded_union_depends_on_the_fallback_being_right():
    """The clause used to read `g3_raw and target == target`. It reduces to
    `g3_raw`, so the numbers were never wrong — but a condition that cannot
    be false is indistinguishable from a guard that silently stopped
    guarding, and no test could see it. Pinned in both directions."""
    right = benchmark_ensemble_router.summarise(
        [_case("7 days", "nope", guarded="nope", fallback="7 days")],
        errors=0, elapsed_s=1.0, router_metrics={}, primary="p", fallback="f")
    wrong = benchmark_ensemble_router.summarise(
        [_case("7 days", "nope", guarded="nope", fallback="still nope")],
        errors=0, elapsed_s=1.0, router_metrics={}, primary="p", fallback="f")
    assert right["metrics"]["union_guarded"] == 1
    assert wrong["metrics"]["union_guarded"] == 0


def test_a_guarded_primary_answer_counts_without_the_fallback():
    report = benchmark_ensemble_router.summarise(
        [_case("10 days", "10", guarded="10 days", fallback="nope")],
        errors=0, elapsed_s=1.0, router_metrics={}, primary="p", fallback="f")
    assert report["metrics"]["gen5_guarded"] == 1
    assert report["metrics"]["union_guarded"] == 1
    assert report["metrics"]["union_raw"] == 0     # raw was still wrong


def test_primary_errors_are_counted_and_not_scored():
    report = benchmark_ensemble_router.summarise(
        [_case("5 days", "5 days")], errors=3, elapsed_s=1.0,
        router_metrics={}, primary="p", fallback="f")
    assert report["metrics"]["n"] == 1
    assert report["metrics"]["errors"] == 3


# --------------------------------------------------------- hetero ensemble

def test_a_failed_fallback_call_no_longer_loses_the_case():
    """`fallback_used` was incremented, then the case was skipped before `n`
    was — so the two counters described different sets of cases and
    `fallback_used` could exceed `n`."""
    observations = [
        {"id": "a", "target": "5 days", "primary_answer": "nope",
         "primary_guarded": "nope", "fallback_error": True},
        {"id": "b", "target": "7 days", "primary_answer": "7 days",
         "primary_guarded": "7 days"},
    ]
    report = benchmark_hetero_ensemble.summarise(
        observations, errors=1, elapsed_s=5.0, primary="p", fallback="f")
    metrics = report["metrics"]
    assert metrics["n"] == 2
    assert metrics["fallback_used"] == 1
    assert metrics["fallback_used"] <= metrics["n"]
    assert metrics["fallback_errors"] == 1
    assert metrics["union_raw"] == 1        # only "b"


def test_the_hetero_union_counts_either_side():
    observations = [
        {"id": "a", "target": "5 days", "primary_answer": "5 days",
         "primary_guarded": "5 days"},
        {"id": "b", "target": "7 days", "primary_answer": "nope",
         "primary_guarded": "nope", "fallback_answer": "7 days"},
    ]
    report = benchmark_hetero_ensemble.summarise(
        observations, errors=0, elapsed_s=5.0, primary="p", fallback="f")
    assert report["metrics"]["union_raw"] == 2
    assert report["metrics"]["primary_raw"] == 1
    assert report["metrics"]["fallback_raw"] == 1


# ---------------------------------------------------------------------------
# The structural property both ensemble benchmarks need, checked across both.
# One of them had it and the other did not, in the same commit — so it is the
# CLASS that needs pinning, not the instance.


@pytest.mark.parametrize("module_name", [
    "benchmark_ensemble_router",
    "benchmark_hetero_ensemble",
    "benchmark_reflex_dispatch",
])
def test_every_model_call_in_a_benchmark_run_is_guarded(module_name):
    """`VllmReplicaRouter.completion` raises RuntimeError once every replica
    has failed. An unguarded call aborts the run and discards every
    observation gathered so far — the evidence file is never written, and the
    gate reports "no evidence" where "evidence says no" belongs.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "research" /
              f"{module_name}.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))

    def catches_runtime_error(handler_node):
        for handler in handler_node.handlers:
            if handler.type is None:
                return True
            names = [handler.type] if not isinstance(handler.type, ast.Tuple) \
                else list(handler.type.elts)
            if any(getattr(n, "id", "") == "RuntimeError" for n in names):
                return True
        return False

    model_call_names = {"ask", "ask_raw", "dispatch", "invoke"}

    def call_name(call):
        return (getattr(call.func, "id", "")
                or getattr(call.func, "attr", ""))

    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}

    def enclosing_function(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return None

    guarded_calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and catches_runtime_error(node):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and call_name(inner) in model_call_names):
                    guarded_calls.add(inner.lineno)

    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and call_name(node) in model_call_names]
    all_calls = {node.lineno for node in calls}
    # benchmark_reflex_dispatch's ask_raw calls are behind its dispatch
    # callback. Both ways to enter that callback (invoke and direct dispatch)
    # must be guarded before those inner calls are considered guarded.
    entry_calls = [node for node in calls
                   if call_name(node) in {"dispatch", "invoke"}]
    if entry_calls and all(node.lineno in guarded_calls for node in entry_calls):
        guarded_calls.update(node.lineno for node in calls
                             if call_name(node) == "ask_raw"
                             and enclosing_function(node) == "dispatch")
    assert all_calls, f"no model call found in {module_name}"
    unguarded = sorted(all_calls - guarded_calls)
    assert not unguarded, (
        f"{module_name}: ask() called without a RuntimeError guard at "
        f"line(s) {unguarded}")


def test_a_failed_fallback_in_the_router_benchmark_is_counted_not_fatal():
    observations = [
        _case("5 days", "nope", guarded="nope") | {"fallback_error": True},
        _case("7 days", "7 days"),
    ]
    report = benchmark_ensemble_router.summarise(
        observations, errors=1, elapsed_s=1.0, router_metrics={},
        primary="p", fallback="f")
    metrics = report["metrics"]
    assert metrics["n"] == 2                 # the case survived
    assert metrics["fallback_errors"] == 1
    assert metrics["fallback_used"] == 1
    assert metrics["fallback_used"] <= metrics["n"]
    assert metrics["union_raw"] == 1         # only the second case
