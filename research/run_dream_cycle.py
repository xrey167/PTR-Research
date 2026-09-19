"""Dream cycle: build the history pool from real generation outcomes, verify
the replay simulator reproduces recorded history (backtest), dream candidate
Gen-7 curriculum policies, and record the winning strategy.

Dreams are off-policy estimates — the actual Gen-7 training run (online
phase) remains the only evidence-producing step, and the gate stays the
only promoter.
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
    if backtest["max_abs_error"] > 0.15:
        raise SystemExit("replay simulator too unfaithful to recorded history — aborting dream")

    candidates = [
        {"concept_oversample": 3, "lookup_anchor": 1},                      # Gen-5/6 shape
        {"concept_oversample": 3, "lookup_anchor": 1, "base_model": "neohorse"},
        {"concept_oversample": 5, "lookup_anchor": 1},
        {"concept_oversample": 3, "lookup_anchor": 2},
        {"concept_oversample": 6, "lookup_anchor": 2},
    ]
    ranked = sim.dream(candidates)
    output = {"status": "dream_cycle_completed",
              "pool_generations": [g["name"] for g in pool.generations],
              "backtest": backtest,
              "ranking": ranked,
              "winner": ranked[0]}
    out_path = PROJECT / "research" / "runs" / "dream-cycle-20260920.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"winner": ranked[0], "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
