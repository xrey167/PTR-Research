"""Dream pod: RSI by dreaming over the generation history (Dream-RSI pattern).

The history pool collects per-case outcomes from every frozen reader eval
report together with the curriculum decisions recorded in each generation's
protocol (which families were oversampled, anchor factors). The replay
simulator turns the recorded (decision -> family delta) pairs into an
off-policy outcome model; candidate curriculum policies are scored by
dreaming over that model without running any training.

Dreams are off-policy estimates, never evidence: only a real training run
(online phase) produces load-bearing outcomes, and only the architecture
gate promotes. The frozen dev/test splits are never touched by dreaming.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

FAMILIES = ("typed", "concept")


def _family(case_id: str, evidence: Any) -> str:
    return "typed" if evidence else "concept"


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
            family = _family(row["id"], case.get("evidence"))
            outcome[family][1] += 1
            outcome[family][0] += 1 if row.get("exact_target_match") else 0
        pool_hash = hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()[:12]
        self.generations.append({
            "name": name, "split": split, "decisions": decisions,
            "outcome": {k: {"correct": v[0], "total": v[1],
                            "rate": v[0] / v[1] if v[1] else 0.0}
                        for k, v in outcome.items()},
            "pool_hash": pool_hash,
        })

    @classmethod
    def from_project(cls, project_root: Path, split: str = "test") -> "HistoryPool":
        """Collect outcomes from every known generation's eval reports.

        Decisions are read from the prepare_generation* intent recorded in
        each protocol's purpose/status; the per-case rows come from the
        adapter eval reports at the gate paths.
        """
        pool = cls()
        known = [
            ("gen3", "runs/qwen3b-eval-test-adapter-20260917-report.json",
             "runs/qwen3b-eval-test-adapter-20260917/cases.json",
             {"concept_oversample": 0, "lookup_anchor": 0}),
            ("gen4", "runs/qwen3b-eval-test-gen4-adapter-20260917-report.json",
             "runs/qwen3b-eval-test-gen4-adapter-20260917/cases.json",
             {"concept_oversample": 3, "lookup_anchor": 0}),
            ("gen5", "runs/qwen3b-eval-test-gen5-20260919-report.json",
             "runs/qwen3b-eval-test-gen5-20260919/cases.json",
             {"concept_oversample": 3, "lookup_anchor": 1}),
            ("gen6", "runs/qwen3b-eval-test-gen6-20260919-report.json",
             "runs/qwen3b-eval-test-gen6-20260919/cases.json",
             {"concept_oversample": 3, "lookup_anchor": 1, "base_model": "neohorse"}),
        ]
        for name, report_path, cases_path, decisions in known:
            report_file = project_root / report_path
            cases_file = project_root / cases_path
            if not report_file.exists() or not cases_file.exists():
                continue
            report = json.loads(report_file.read_text(encoding="utf-8"))
            cases = json.loads(cases_file.read_text(encoding="utf-8"))
            pool.add_generation(name, decisions, split, report, cases)
        if len(pool.generations) < 2:
            raise ValueError("history pool needs at least two generations to dream over")
        return pool

    def family_deltas(self) -> dict[str, dict[str, float]]:
        """Recorded per-decision family rate changes between consecutive generations."""
        deltas: dict[str, dict[str, float]] = {"concept_oversample": {}, "lookup_anchor": {}}
        for prev, curr in zip(self.generations, self.generations[1:]):
            for family in FAMILIES:
                delta = curr["outcome"][family]["rate"] - prev["outcome"][family]["rate"]
                for decision, level in curr["decisions"].items():
                    if decision not in deltas:
                        continue
                    prev_level = prev["decisions"].get(decision, 0)
                    if level != prev_level:
                        step = delta / max(level - prev_level, 1)
                        deltas[decision][family] = step
        return deltas


class ReplaySimulator:
    """Off-policy outcome model: linear response of family rates to decisions."""

    def __init__(self, pool: HistoryPool):
        self.pool = pool
        self.baseline = {family: pool.generations[0]["outcome"][family]["rate"]
                         for family in FAMILIES}
        self.responses = pool.family_deltas()
        # Per-decision limits learned from history (never dream beyond it).
        self.limits = {"concept_oversample": (0, 6), "lookup_anchor": (0, 2)}

    def simulate(self, decisions: dict[str, Any]) -> dict[str, dict[str, float]]:
        out = {}
        for family in FAMILIES:
            rate = self.baseline[family]
            for decision, level in decisions.items():
                if decision in self.responses and family in self.responses[decision]:
                    rate += self.responses[decision][family] * level
            out[family] = {"predicted_rate": round(min(rate, 1.0), 4)}
        return out

    def dream(self, policies: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Evaluate candidate curriculum policies off-policy; returns ranking."""
        ranked = []
        for decisions in policies:
            for decision, (low, high) in self.limits.items():
                value = decisions.get(decision, 0)
                if not low <= value <= high:
                    raise ValueError(f"policy exceeds learned limits: {decision}={value}")
            prediction = self.simulate(decisions)
            score = sum(prediction[f]["predicted_rate"] for f in FAMILIES)
            ranked.append({"decisions": decisions, "prediction": prediction,
                           "score": round(score, 4),
                           "pool_hash": self.pool.generations[-1]["pool_hash"]})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def backtest(self) -> dict[str, Any]:
        """Faithfulness check: replay every recorded generation and report error."""
        errors = []
        for generation in self.pool.generations:
            prediction = self.simulate(generation["decisions"])
            for family in FAMILIES:
                actual = generation["outcome"][family]["rate"]
                predicted = prediction[family]["predicted_rate"]
                errors.append(abs(predicted - actual))
        return {"generations": len(self.pool.generations),
                "mean_abs_error": round(sum(errors) / max(len(errors), 1), 4),
                "max_abs_error": round(max(errors), 4)}
