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

The Dream-Pod design names three obligations besides the replay itself, and
this runner now meets all three: the cycle takes a RAM lease from a
ResourceGovernor (replay is data analysis, but it is not free), it consumes
an autonomy quota counted in the event log, and it writes a `dream_cycle`
provenance event carrying the pool fingerprint, every candidate and the
winner. Before, the cycle left nothing behind but a JSON file.

Two details decide whether those obligations bind anything:

  * The RAM budget comes from `ResourceBudget.from_snapshot(probe_hardware())`,
    i.e. from the machine, not from a number the caller passed in. A governor
    constructed inside this function against a caller-named budget could not
    refuse, which would make the lease a receipt the cycle writes to itself.
    `--ram-budget-bytes` still overrides it, for a deliberately tighter bound.
  * The `dream_cycle` event is written on EVERY exit path, not only when a
    winner was found. An aborting cycle has already read the pool, run the
    backtest and dreamed every candidate; if it left no event behind it could
    be repeated without limit while the quota kept reading "0 used".
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.dream import (CycleBudget, HistoryPool, ReplaySimulator,
                               record_cycle)
from neural_pods.registry import Registry
from neural_pods.resource_runtime import (ResourceBudget, ResourceGovernor,
                                          probe_hardware)

PROJECT = Path(__file__).resolve().parents[1]
REPLAY_RAM_BYTES = 256 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path,
                        default=PROJECT / "runs" / "dream-registry.sqlite3",
                        help="provenance log the cycle is recorded in")
    parser.add_argument("--max-cycles-per-day", type=int, default=8)
    parser.add_argument("--ram-budget-bytes", type=int, default=None,
                        help="override the RAM budget the lease is taken "
                             "against; by default it comes from the machine")
    args = parser.parse_args()

    args.registry.parent.mkdir(parents=True, exist_ok=True)
    registry = Registry(args.registry)
    budget = CycleBudget(registry, max_per_window=args.max_cycles_per_day)
    # Checked BEFORE the pool is read: an exhausted quota must not first
    # spend the resources it is meant to bound.
    budget.check()
    print(json.dumps({"autonomy_budget": budget.stats()}, indent=2))

    if args.ram_budget_bytes is None:
        machine = ResourceBudget.from_snapshot(probe_hardware(path=str(PROJECT)))
        ram_budget = machine.ram_bytes
        budget_source = "machine snapshot"
    else:
        ram_budget = int(args.ram_budget_bytes)
        budget_source = "--ram-budget-bytes"
    governor = ResourceGovernor(
        ResourceBudget(ram_bytes=ram_budget, vram_bytes={}, disk_bytes=None))
    print(json.dumps({"ram_budget_bytes": ram_budget,
                      "ram_budget_source": budget_source}, indent=2))
    lease = governor.try_acquire("ram", REPLAY_RAM_BYTES)
    if lease is None:
        raise SystemExit(
            f"no RAM lease available for the replay ({REPLAY_RAM_BYTES} bytes "
            f"against a {ram_budget} byte budget from {budget_source}) "
            "- aborting dream")

    lease_payload = {"tier": lease.tier, "amount_bytes": lease.amount_bytes,
                     "budget_bytes": ram_budget, "budget_source": budget_source}
    # `progress` is what the cycle got as far as. It is filled in as the cycle
    # advances so that the `finally` below can record an aborted cycle with
    # everything it already produced.
    progress: dict = {"status": "aborted", "reason": None, "recorded": False}
    try:
        _cycle(pool_root=PROJECT, registry=registry, budget=budget,
               lease=lease_payload, progress=progress)
    except SystemExit as stop:
        progress["reason"] = str(stop) or f"SystemExit({stop.code!r})"
        raise
    except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
        progress["reason"] = repr(error)
        raise
    finally:
        governor.release(lease)
        if not progress["recorded"]:
            aborted = record_cycle(
                registry, pool=progress.get("pool"), ranked=progress.get("ranked"),
                winner=None, backtest=progress.get("backtest"),
                lease=lease_payload, status=progress["status"],
                reason=progress["reason"])
            print(json.dumps({"dream_cycle_recorded_as": aborted["status"],
                              "reason": aborted["reason"],
                              "autonomy_budget": budget.stats()}, indent=2))


def _cycle(*, pool_root: Path, registry, budget, lease, progress: dict) -> None:
    pool = HistoryPool.from_project(pool_root, split="test")
    progress["pool"] = pool
    print(json.dumps({"pool": [g["name"] for g in pool.generations],
                      "outcomes": {g["name"]: g["outcome"] for g in pool.generations}}, indent=2))

    sim = ReplaySimulator(pool)
    backtest = sim.backtest()
    progress["backtest"] = backtest
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
    progress["ranked"] = ranked
    if not ranked[0]["estimable"]:
        # Ranking puts estimable policies first, so an unestimable head means
        # the history identifies none of them. Declaring that entry "winner"
        # would hand a decision to a model that has no opinion.
        unidentified = ranked[0]["prediction"]["typed"]["unidentified"]
        raise SystemExit(
            "no candidate policy is identified by the recorded history "
            f"(unidentified: {', '.join(unidentified)}) - record another "
            "generation before dreaming")
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
    provenance = record_cycle(
        registry, pool=pool, ranked=ranked, winner=ranked[0], backtest=backtest,
        lease=lease, status="completed")
    progress["recorded"] = True
    progress["status"] = "completed"
    output["provenance"] = provenance
    output["autonomy_budget"] = budget.stats()
    out_path = PROJECT / "research" / "runs" / "dream-cycle-20260920.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"winner": ranked[0], "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
