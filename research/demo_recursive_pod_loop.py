"""Small, reproducible demonstration of replayable broad-to-deep Pods."""
from __future__ import annotations
import json
from pathlib import Path

from neural_pods.recursive_memory import BroadThenDeepExplorer, DiscoveryReplay, Experience
from neural_pods.pod_types import typed_artifact
from neural_pods.registry import Registry


def main() -> None:
    seeds = [
        Experience("root-login", "browser", "login", "submit", "unknown", False,
                    metadata={"confidence": "0.2"}),
        Experience("root-search", "browser", "search", "query", "results", True,
                    metadata={"confidence": "0.92"}),
    ]
    expanded = {
        "root-login": [Experience("deep-login", "browser", "login", "submit", "ok", True,
                                   depth=1, parent_id="root-login", metadata={"confidence": "0.95"})],
        "root-search": [],
    }
    explorer = BroadThenDeepExplorer(max_depth=2, confidence_threshold=0.75)
    history = explorer.explore(seeds, lambda item: expanded.get(item.experience_id, []))
    replay = DiscoveryReplay(history)
    score = replay.score(lambda condition, action: {"login": "ok", "search": "results"}.get(condition, ""))
    pods = explorer.consolidate(history)

    reg = Registry(":memory:")
    origin = reg.origin("experience-replay", "browser", "1", {"history": [x.experience_id for x in history]})
    knowledge = reg.publish("experience:browser", {"subject": "browser", "predicate": "verified_experience"}, [origin])
    artifact = typed_artifact(reg, "text", "context", pods[0].to_payload(), [knowledge])
    active_before_revoke = bool(reg.snapshot([artifact]).artifacts)
    revoked = reg.revoke(origin)
    blocked_after_revoke = False
    try:
        reg.snapshot([artifact])
    except Exception:
        blocked_after_revoke = True
    report = {
        "schema": "recursive-pod-loop-demo:v1",
        "history_count": len(history),
        "deep_expansions": len(history) - len(seeds),
        "replay": score.__dict__,
        "consolidated_pods": len(pods),
        "lineage_artifact": artifact,
        "active_before_revoke": active_before_revoke,
        "blocked_after_revoke": blocked_after_revoke,
        "revoked_count": len(revoked),
        "passed": len(history) == 3 and score.accuracy == 1.0 and len(pods) == 2 and
                  active_before_revoke and blocked_after_revoke,
        "scope": "local deterministic replay fixture; not a claim about external agent benchmarks",
    }
    out = Path("runs/recursive-pod-loop-001.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    reg.close()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
