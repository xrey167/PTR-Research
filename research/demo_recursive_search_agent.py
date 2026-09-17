"""Run the real local iterative search agent through broad/deep memory."""
from __future__ import annotations
import json
from pathlib import Path
from neural_pods.local_search import LocalSearchBackend
from neural_pods.recursive_search import RecursiveSearchRunner


def main():
    root = Path("runs/recursive-search-agent-001")
    root.mkdir(parents=True, exist_ok=True)
    backend = LocalSearchBackend(root / "namespace.sqlite")
    backend.create_namespace("corpus")
    backend.upsert("corpus", "doc-a", text="Phil Redmond created Brookside", vector=[1, 0], metadata={"status": "active"})
    backend.upsert("corpus", "doc-b", text="Brookside premiered on Channel 4", vector=[.9, .1], metadata={"status": "active"})
    backend.upsert("corpus", "noise", text="unrelated document", vector=[0, 1], metadata={"status": "active"})
    result = RecursiveSearchRunner(backend, "corpus", max_deep_turns=2).run(
        "Which documents explain Brookside and Channel 4?", ["doc-a", "doc-b"])
    report = {
        "schema": "recursive-search-agent-demo:v1",
        "broad_recall": result.broad.recall,
        "deep_recall": result.deep.recall if result.deep else None,
        "broad_search_calls": result.broad.search_calls,
        "deep_search_calls": result.deep.search_calls if result.deep else 0,
        "experience_records": len(result.experiences),
        "verified_records": sum(x.verified for x in result.experiences),
        "experience_pods": [x.pod_id for x in result.pods],
        "passed": result.broad.recall < 1.0 and result.deep is not None and result.deep.recall == 1.0 and bool(result.pods),
        "scope": "local three-document fixture; deterministic search policy, not an external agent benchmark",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    backend.close()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
