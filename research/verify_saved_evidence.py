"""Check invariants of the persisted evidence artifacts used by the project."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    demo = load("runs/assembled-stack-demo-001.json")
    taxonomy = load("runs/pod-taxonomy-balance-001/report.json")
    neo = load("runs/neohorse-real-checkpoint-probe-001.json")
    local = load("runs/local-stack-benchmark-002/report.json")
    checks = {
        "showcase_three_pods": demo["pods"] == 3 and all(
            row["route_matches_type"] for row in demo["queries"]
        ),
        "showcase_revocation": demo["revocation_removed_lead_time"] is True,
        "taxonomy_balanced": taxonomy["balanced"] is True,
        "taxonomy_metrics_present": taxonomy["dragonfly_test_top1"] > 0,
        "neohorse_real_throughput": neo["total_generated_tokens"] > 0 and neo["mean_tokens_per_s"] > 0,
        "local_retrieval_recall": local["episode_recall"] == 1.0 and local["episode_mrr"] == 1.0,
    }
    report = {"schema": "saved-evidence-verification:v1", "passed": all(checks.values()), "checks": checks}
    path = ROOT / "runs" / "saved-evidence-verification-001.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
