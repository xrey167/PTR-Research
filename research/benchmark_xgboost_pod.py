"""P2 of the Pod-Arm design: an XGBoost pod behind the executor contract.

The design's phase table promised a `xgboost_pod` gate check for this phase
and marked the phase deliverable. The check never existed, so what actually
held P2 up was a unit test of the factory. This benchmark is that check's
evidence, and it runs on any machine — xgboost is a hard dependency, nothing
here needs a GPU, a broker or the LXD nodes.

What it measures, each of which is one thing the phase claims:

  1. REGISTRATION — a runtime is one registration away from being usable:
     the pod is created through `ExecutorFactory`, never by calling
     `TreeExecutor` directly.
  2. THE LEASE IS REAL — activation goes through `GovernedExecutorPool` with
     a genuine `ResourceGovernor`, and the budget refuses an activation that
     does not fit. A budget that cannot refuse is not a budget, which is the
     defect this same audit found in the dream cycle's self-issued lease.
  3. DETERMINISM — the same input twice, in two independently activated
     pods, produces bit-identical output. A pod whose answer depends on when
     it was loaded cannot be verified by the guard.
  4. RELEASE ACCOUNTS — after release the governor is back at zero. A
     restart that leaks its predecessor's bytes shrinks the budget with
     every cycle, which is what `release()` used to do.

The model is trained here, from a fixed seed, so the evidence is
reproducible from the repository alone and no checkpoint has to be shipped.
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.pod_executor import (ExecutorFactory, GovernedExecutorPool,
                                      TreeExecutor)  # noqa: E402
from neural_pods.resource_runtime import (ResourceBudget,  # noqa: E402
                                          ResourceGovernor)
from research.evidence import write as write_evidence  # noqa: E402

#: The modules these numbers are evidence ABOUT.
SUBJECT = [
    "neural_pods/pod_executor.py",
    "neural_pods/resource_runtime.py",
]

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "research" / "runs" / "xgboost-pod-20260920.json"
POD_RAM_BYTES = 64 * 1024 * 1024
SEED = 20260920
ROWS = 512
FEATURES = 6


def _train_model(path: Path) -> dict:
    """A small, fully determined classifier. The task is deliberately easy:
    this benchmark is about the pod contract, not about model quality, and a
    model that predicts noise would make the determinism check vacuous."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(ROWS, FEATURES))
    # A rule the trees can learn exactly, so the probabilities are stable.
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3,
         "seed": SEED, "nthread": 1},
        xgb.DMatrix(x, label=y), num_boost_round=24)
    booster.save_model(str(path))
    return {"rows": ROWS, "features": FEATURES, "seed": SEED,
            "rounds": 24, "model_bytes": path.stat().st_size}


def _probe_rows() -> list[list[float]]:
    import numpy as np
    rng = np.random.default_rng(SEED + 1)
    return [list(map(float, row)) for row in rng.normal(size=(32, FEATURES))]


def measure() -> dict:
    """The measurement, as a function that returns its result.

    Separated from `main()` on purpose. 85 of this repository's benchmark
    scripts are `if __name__ == "__main__"` monoliths, which is why 6.5k
    lines of measurement code have no test between them and the gate — and
    why five of the eight defects the 2026-09-20 review found sat in exactly
    that layer. A measurement that can be called can be checked against a
    known answer; this one is, in tests/test_benchmark_xgboost_pod.py.
    """
    with tempfile.TemporaryDirectory() as workdir:
        model_path = Path(workdir) / "tree-pod.json"
        training = _train_model(model_path)

        factory = ExecutorFactory()
        factory.register(
            "xgboost", lambda: TreeExecutor(model_path, ram_bytes=POD_RAM_BYTES))

        # (2) A budget with room for exactly two pods, so the third refusal
        #     is a fact about the budget and not about the machine.
        governor = ResourceGovernor(ResourceBudget(
            ram_bytes=2 * POD_RAM_BYTES, vram_bytes={}, disk_bytes=None))
        pool = GovernedExecutorPool(factory, governor=governor)

        first = pool.activate("tree-pod-a", "xgboost")
        second = pool.activate("tree-pod-b", "xgboost")

        refused = None
        try:
            pool.activate("tree-pod-c", "xgboost")
        except RuntimeError as error:
            refused = str(error)

        # (3) Two independently activated pods, same inputs.
        rows = _probe_rows()
        first_answers = [first.infer({"features": row}) for row in rows]
        second_answers = [second.infer({"features": row}) for row in rows]
        repeat_answers = [first.infer({"features": row}) for row in rows]

        # ResourceGovernor.stats() reports `used` keyed by "tier:device" and
        # omits a tier at zero. Read explicitly rather than through a default:
        # `(stats.get("used_bytes") or 0) == 0` would have been satisfied by a
        # key that does not exist, which is the same defect this audit found
        # in the gate's `or 999` clauses.
        def ram_used(stats: dict) -> int:
            return int(stats.get("used", {}).get("ram:None", 0))

        governor_at_peak = ram_used(governor.stats())

        # (4) Release accounts for what was acquired.
        pool.release("tree-pod-a")
        pool.release("tree-pod-b")
        governor_after = ram_used(governor.stats())
        # A pod id may be re-activated once its predecessor is gone; the
        # double-activation leak used to make this permanently impossible.
        reactivated = pool.activate("tree-pod-a", "xgboost")
        reactivated_ok = reactivated.health()
        pool.release("tree-pod-a")

        result = {
            "status": "completed",
            "training": training,
            "probe_rows": len(rows),
            "registered_kinds": list(factory.kinds()),
            # (1) created through the factory, by kind, never by class
            "created_via_factory": first.runtime == "xgboost",
            # (2) the budget refused a third pod it had no room for
            "budget_refused_third_pod": refused is not None,
            "budget_refusal": refused,
            "ram_budget_bytes": 2 * POD_RAM_BYTES,
            "pod_ram_bytes": POD_RAM_BYTES,
            "governor_used_bytes_at_peak": governor_at_peak,
            "peak_matches_two_pods": governor_at_peak == 2 * POD_RAM_BYTES,
            # (3) determinism, across pods and across calls
            "deterministic_across_pods": first_answers == second_answers,
            "deterministic_across_calls": first_answers == repeat_answers,
            "distinct_predictions": len({a["prediction"] for a in first_answers}),
            # (4) release gives the bytes back, and the id becomes reusable
            "governor_used_bytes_after_release": governor_after,
            "released_to_zero": governor_after == 0,
            "reactivation_after_release": bool(reactivated_ok),
            "pool_stats": pool.stats(),
            "sample_answer": first_answers[0],
        }
        return result


def main() -> None:
    result = measure()
    write_evidence(result, OUT, __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
