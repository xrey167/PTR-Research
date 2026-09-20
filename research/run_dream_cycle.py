"""Dream cycle: build the history pool from real generation outcomes, measure
how far the replay simulator can actually predict (leave-one-generation-out
backtest), dream candidate curriculum policies, and record the winner.

Dreams are off-policy estimates — the actual training run (online phase)
remains the only evidence-producing step, and the gate stays the only
promoter.

The backtest no longer asks whether the model reproduces the points it was
fitted on: with one intercept plus one coefficient per decision and one
decision change per generation it always does, which is why the earlier
"backtest error 0.01" carried no information. What it asks now is whether a
generation can be predicted from the OTHERS, and it reports the generations
where the remaining history does not identify the decisions involved.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.dream import HistoryPool, ReplaySimulator

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    pool = HistoryPool.from_project(PROJECT, split="test")
    print(json.dumps({"pool": [g["name"] for g in pool.generations],
                      "outcomes": {g["name"]: g["outcome"] for g in pool.generations}}, indent=2))

    sim = ReplaySimulator(pool)
    backtest = sim.backtest()
    print(json.dumps({"backtest": backtest}, indent=2))
    out_of_sample = backtest["out_of_sample"]
    if out_of_sample["generations"] and out_of_sample["max_abs_error"] > 0.15:
        raise SystemExit(
            "replay simulator failed out-of-sample backtest "
            f"({out_of_sample['max_abs_error']}) — aborting dream")

    candidates = [
        {"concept_oversample": 3, "lookup_anchor": 1},                      # Gen-5/6 shape
        {"concept_oversample": 3, "lookup_anchor": 1, "base_model": "neohorse"},
        {"concept_oversample": 5, "lookup_anchor": 1},
        {"concept_oversample": 3, "lookup_anchor": 2},
        {"concept_oversample": 6, "lookup_anchor": 2},
    ]
    # Some candidates deliberately go past the levels the history showed
    # (lookup_anchor 2 was never observed). That is allowed, but every entry
    # carries an `extrapolates` list saying exactly where it left the data.
    ranked = sim.dream(candidates, allow_extrapolation=True)
    output = {"status": "dream_cycle_completed",
              "pool_generations": [g["name"] for g in pool.generations],
              "backtest": backtest,
              "predictive_validity": (
                  "none measured: no generation could be predicted from the "
                  "others" if not out_of_sample["generations"]
                  else f"{out_of_sample['generations']} generation(s) predicted "
                       f"out of sample, max abs error "
                       f"{out_of_sample['max_abs_error']}"),
              "ranking": ranked,
              "winner": ranked[0]}
    out_path = PROJECT / "research" / "runs" / "dream-cycle-20260920.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"winner": ranked[0], "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
