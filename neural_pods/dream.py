"""Dream pod: RSI by dreaming over the generation history (Dream-RSI pattern).

The history pool collects per-case outcomes from every frozen reader eval
report together with the curriculum decisions recorded in each generation's
protocol (which families were oversampled, anchor factors, which base model).
The replay simulator turns the recorded (decision -> family delta) pairs into
an off-policy outcome model; candidate curriculum policies are scored by
dreaming over that model without running any training.

Dreams are off-policy estimates, never evidence: only a real training run
(online phase) produces load-bearing outcomes, and only the architecture
gate promotes. The frozen dev/test splits are never touched by dreaming.

How far the estimates can be trusted is now something this module reports
instead of assuming. Three properties matter:

  * The model is an intercept plus one coefficient per decision. The history
    changes one decision per generation, so in-sample the parameters
    reproduce the very points they were fitted on — residuals near zero there
    mean nothing at all. `backtest()["in_sample"]["degenerate"]` says so.
  * `backtest()` is therefore leave-one-generation-out. A generation can only
    be predicted out of sample when the remaining transitions still identify
    every decision it uses; when they do not, it is skipped WITH a reason
    rather than silently counted.
  * A transition that moves two decisions at once attributes to neither.
    Those are collected in `ambiguous` instead of being charged to whichever
    decision happened to be iterated last.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

FAMILIES = ("typed", "concept")


def _family(evidence: Any) -> str:
    return "typed" if evidence else "concept"


def _levels(decisions: dict[str, Any]) -> dict[str, float]:
    """Decision values as numbers.

    A non-numeric value becomes a 0/1 indicator under the key
    "<name>=<value>". That is what turns the Gen-5 -> Gen-6 base-model swap
    into a modelled dimension; previously it was not in the delta table at
    all and its whole effect landed in the residual.
    """
    levels: dict[str, float] = {}
    for name, value in decisions.items():
        if isinstance(value, bool):
            levels[name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            levels[name] = float(value)
        else:
            levels[f"{name}={value}"] = 1.0
    return levels


class HistoryPool:
    """Append-only pool of generation records: decision + per-family outcome."""

    def __init__(self):
        self.generations: list[dict[str, Any]] = []

    def add_generation(self, name: str, decisions: dict[str, Any],
                       split: str, report: dict[str, Any], cases: list[dict]) -> None:
        cases_by_id = {c["id"]: c for c in cases}
        rows = report.get("rows", [])
        outcome = {family: [0, 0] for family in FAMILIES}
        for row in rows:
            case = cases_by_id.get(row["id"])
            if case is None:
                continue
            family = _family(case.get("evidence"))
            outcome[family][1] += 1
            outcome[family][0] += 1 if row.get("exact_target_match") else 0
        pool_hash = hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()[:12]
        self.generations.append({
            "name": name, "split": split, "decisions": decisions,
            "levels": _levels(decisions),
            "outcome": {k: {"correct": v[0], "total": v[1],
                            "rate": v[0] / v[1] if v[1] else 0.0}
                        for k, v in outcome.items()},
            "pool_hash": pool_hash,
        })

    @classmethod
    def from_project(cls, project_root: Path, split: str = "test") -> "HistoryPool":
        """Collect outcomes from every known generation's eval reports.

        Each file is looked for under research/runs/ first (checked in) and
        then under runs/ (the server's output, which .gitignore excludes), so
        a clone that carries the reports can rebuild the pool.
        """
        pool = cls()
        known = [
            ("gen3", "qwen3b-eval-test-adapter-20260917-report.json",
             "qwen3b-eval-test-adapter-20260917/cases.json",
             {"concept_oversample": 0, "lookup_anchor": 0}),
            ("gen4", "qwen3b-eval-test-gen4-adapter-20260917-report.json",
             "qwen3b-eval-test-gen4-adapter-20260917/cases.json",
             {"concept_oversample": 3, "lookup_anchor": 0}),
            ("gen5", "qwen3b-eval-test-gen5-20260919-report.json",
             "qwen3b-eval-test-gen5-20260919/cases.json",
             {"concept_oversample": 3, "lookup_anchor": 1}),
            ("gen6", "qwen3b-eval-test-gen6-20260919-report.json",
             "qwen3b-eval-test-gen6-20260919/cases.json",
             {"concept_oversample": 3, "lookup_anchor": 1, "base_model": "neohorse"}),
        ]
        search = [project_root / "research" / "runs", project_root / "runs"]
        skipped = []
        for name, report_name, cases_name, decisions in known:
            report_file = next((d / report_name for d in search
                                if (d / report_name).exists()), None)
            cases_file = next((d / cases_name for d in search
                               if (d / cases_name).exists()), None)
            if report_file is None or cases_file is None:
                skipped.append(name)
                continue
            pool.add_generation(
                name, decisions, split,
                json.loads(report_file.read_text(encoding="utf-8")),
                json.loads(cases_file.read_text(encoding="utf-8")))
        if len(pool.generations) < 2:
            raise ValueError(
                "history pool needs at least two generations to dream over; "
                f"missing evidence for: {', '.join(skipped) or 'unknown'}")
        pool.skipped = skipped
        return pool

    def family_deltas(self, *, exclude: str | None = None
                      ) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
        """Per-decision family response, averaged over every clean transition.

        Returns (responses, ambiguous). A transition in which more than one
        decision moved cannot be attributed to any single one of them and is
        returned in `ambiguous` rather than charged to one.
        """
        generations = [g for g in self.generations if g["name"] != exclude]
        slopes: dict[str, dict[str, list[float]]] = {}
        ambiguous: list[dict[str, Any]] = []
        for prev, curr in zip(generations, generations[1:]):
            names = set(prev["levels"]) | set(curr["levels"])
            moved = {name for name in names
                     if curr["levels"].get(name, 0.0) != prev["levels"].get(name, 0.0)}
            if not moved:
                continue
            if len(moved) > 1:
                ambiguous.append({"from": prev["name"], "to": curr["name"],
                                  "decisions": sorted(moved)})
                continue
            decision = moved.pop()
            step = curr["levels"].get(decision, 0.0) - prev["levels"].get(decision, 0.0)
            for family in FAMILIES:
                delta = curr["outcome"][family]["rate"] - prev["outcome"][family]["rate"]
                slopes.setdefault(decision, {}).setdefault(family, []).append(delta / step)
        responses = {decision: {family: sum(values) / len(values)
                                for family, values in families.items()}
                     for decision, families in slopes.items()}
        return responses, ambiguous

    def observed_limits(self, *, exclude: str | None = None) -> dict[str, tuple[float, float]]:
        """The level range each decision actually took in the history."""
        generations = [g for g in self.generations if g["name"] != exclude]
        limits: dict[str, tuple[float, float]] = {}
        for generation in generations:
            names = set(limits) | set(generation["levels"])
            for name in names:
                level = generation["levels"].get(name, 0.0)
                low, high = limits.get(name, (level, level))
                limits[name] = (min(low, level), max(high, level))
        return limits


class ReplaySimulator:
    """Off-policy outcome model: linear response of family rates to decisions."""

    def __init__(self, pool: HistoryPool, *, exclude: str | None = None):
        self.pool = pool
        self.exclude = exclude
        self.generations = [g for g in pool.generations if g["name"] != exclude]
        if not self.generations:
            raise ValueError("nothing left in the pool after exclusion")
        self.responses, self.ambiguous = pool.family_deltas(exclude=exclude)
        self.limits = pool.observed_limits(exclude=exclude)
        self.baseline, self.baseline_source = self._fit_baseline()

    def _fit_baseline(self) -> tuple[dict[str, float] | None, str | None]:
        """Intercept = the generation that took every decision at level 0.

        Without such a generation the intercept is not identified from the
        remaining data, and the simulator says so instead of borrowing the
        first row it can find.
        """
        for generation in self.generations:
            if all(level == 0.0 for level in generation["levels"].values()):
                return ({family: generation["outcome"][family]["rate"]
                         for family in FAMILIES}, generation["name"])
        return None, None

    def simulate(self, decisions: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Predicted family rates, with whether the prediction is identified."""
        levels = _levels(decisions)
        unknown = sorted(name for name, level in levels.items()
                         if level != 0.0 and name not in self.responses)
        estimable = self.baseline is not None and not unknown
        out: dict[str, dict[str, Any]] = {}
        for family in FAMILIES:
            if not estimable:
                out[family] = {"predicted_rate": None, "estimable": False,
                               "unidentified": unknown or ["intercept"]}
                continue
            rate = self.baseline[family]
            for name, level in levels.items():
                response = self.responses.get(name, {})
                if family in response:
                    rate += response[family] * level
            out[family] = {"predicted_rate": round(min(max(rate, 0.0), 1.0), 4),
                           "estimable": True, "unidentified": []}
        return out

    def extrapolation(self, decisions: dict[str, Any]) -> list[dict[str, Any]]:
        """Levels asked for that the history never showed."""
        beyond = []
        for name, level in _levels(decisions).items():
            low, high = self.limits.get(name, (0.0, 0.0))
            if level < low or level > high:
                beyond.append({"decision": name, "level": level,
                               "observed": [low, high]})
        return beyond

    def dream(self, policies: Iterable[dict[str, Any]], *,
              allow_extrapolation: bool = False) -> list[dict[str, Any]]:
        """Score candidate curriculum policies off-policy; returns a ranking.

        `allow_extrapolation=False` is the documented intent — never dream
        beyond the levels the history actually showed. The previous hard-coded
        limit table permitted lookup_anchor=2 although only 0 and 1 were ever
        observed, which is how the winning Gen-7 policy came to be an
        extrapolation without anyone noticing. Extrapolating on purpose is
        fine; it is now explicit and flagged in every entry.
        """
        ranked = []
        for decisions in policies:
            beyond = self.extrapolation(decisions)
            if beyond and not allow_extrapolation:
                raise ValueError(f"policy goes beyond observed history: {beyond}")
            prediction = self.simulate(decisions)
            estimable = all(prediction[family]["estimable"] for family in FAMILIES)
            score = (round(sum(prediction[f]["predicted_rate"] for f in FAMILIES), 4)
                     if estimable else None)
            ranked.append({"decisions": decisions, "prediction": prediction,
                           "score": score, "estimable": estimable,
                           "extrapolates": beyond,
                           "pool_hash": self.generations[-1]["pool_hash"]})
        ranked.sort(key=lambda item: (item["score"] is not None, item["score"] or 0.0),
                    reverse=True)
        return ranked

    def backtest(self) -> dict[str, Any]:
        """Leave-one-generation-out, with the in-sample number for contrast.

        The in-sample residuals are reported but marked: with one intercept
        plus one coefficient per decision and one decision change per
        generation, the fit reproduces its own fitting points, so a small
        in-sample error is arithmetic, not evidence. `degenerate` is true
        exactly when the model has at least as many free parameters as the
        history has generations.
        """
        in_sample_errors = []
        for generation in self.generations:
            prediction = self.simulate(generation["decisions"])
            for family in FAMILIES:
                if prediction[family]["estimable"]:
                    in_sample_errors.append(abs(
                        prediction[family]["predicted_rate"]
                        - generation["outcome"][family]["rate"]))
        parameters = 1 + len(self.responses)

        out_errors: list[float] = []
        per_generation = []
        for generation in self.pool.generations:
            held_out = generation["name"]
            try:
                reduced = ReplaySimulator(self.pool, exclude=held_out)
            except ValueError:
                per_generation.append({"generation": held_out, "predicted": False,
                                       "reason": "pool empty without it"})
                continue
            prediction = reduced.simulate(generation["decisions"])
            if not all(prediction[f]["estimable"] for f in FAMILIES):
                per_generation.append({
                    "generation": held_out, "predicted": False,
                    "reason": "not identified without it: "
                              + ", ".join(prediction[FAMILIES[0]]["unidentified"])})
                continue
            errors = {family: round(abs(prediction[family]["predicted_rate"]
                                        - generation["outcome"][family]["rate"]), 4)
                      for family in FAMILIES}
            out_errors.extend(errors.values())
            per_generation.append({"generation": held_out, "predicted": True,
                                   "abs_error": errors})

        return {
            "mode": "leave_one_generation_out",
            "generations": len(self.pool.generations),
            "parameters": parameters,
            "ambiguous_transitions": self.ambiguous,
            "in_sample": {
                "mean_abs_error": round(sum(in_sample_errors)
                                        / max(len(in_sample_errors), 1), 4),
                "max_abs_error": round(max(in_sample_errors, default=0.0), 4),
                "degenerate": parameters >= len(self.generations),
                "note": ("with at least as many parameters as generations the "
                         "fit reproduces its own points; this number is not a "
                         "measure of predictive accuracy"),
            },
            "out_of_sample": {
                "generations": sum(1 for row in per_generation if row["predicted"]),
                "mean_abs_error": (round(sum(out_errors) / len(out_errors), 4)
                                   if out_errors else None),
                "max_abs_error": (round(max(out_errors), 4) if out_errors else None),
                "per_generation": per_generation,
            },
        }
