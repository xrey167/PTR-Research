"""Reproducible local acceptance gate for the assembled research stack.

This intentionally reports mechanism evidence and its limits separately.  It
does not turn synthetic or proxy experiments into a claim about unavailable
private CQP1/J-Space weights.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
VM = ROOT / "research" / "imported_evidence_20260915" / "So_Neural_Knowledge_VM_R211_R214"


def run(cmd: list[str], cwd: Path = ROOT, timeout: int = 600) -> dict:
    env = os.environ.copy()
    # Keep pytest's temporary tree project-local and unique to this gate run.
    # A stale Windows `%TEMP%/pytest-of-*` directory can be ACL-protected after
    # interrupted runs, which otherwise makes the acceptance gate fail in setup.
    tmp = ROOT / ".pytest-gate-tmp"
    tmp.mkdir(exist_ok=True)
    env.update(TEMP=str(tmp), TMP=str(tmp), TMPDIR=str(tmp))
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout)
    return {"command": cmd, "returncode": p.returncode,
            "output_tail": p.stdout[-4000:]}


def main() -> int:
    py = sys.executable
    checks: list[dict] = []
    # The repository is evaluated in locked-down Windows workspaces where
    # pytest's cache directory may be ACL-protected.  The gate only needs the
    # test result, so disable cache writes and keep the run warning-free.
    checks.append(run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "--basetemp", str(ROOT / ".pytest-gate-basetemp")]))
    checks.append(run([py, "research/audit_linked_model.py",
                       "runs/linked-model-replay-002"]))
    checks.append(run([py, "research/audit_replay_revocation.py",
                       "runs/linked-model-replay-002"]))
    checks.append(run([py, "research/audit_replay_generation.py",
                       "runs/linked-model-replay-002"]))
    checks.append(run([py, "research/evaluate_jspace_local.py"]))
    # The expensive GPU run is persisted; validate its measured acceptance
    # contract on every local gate invocation.
    regression = RUNS / "reader-lora-mixed-multihop-001" / "regression-report.json"
    try:
        data = json.loads(regression.read_text(encoding="utf-8"))
        ok = (data.get("two-hop", {}).get("correct") == data.get("two-hop", {}).get("total") == 4
              and data.get("three-hop", {}).get("correct") == data.get("three-hop", {}).get("total") == 4)
        checks.append({"command": ["persisted", str(regression)], "returncode": 0 if ok else 1,
                       "output_tail": "two-hop=4/4, three-hop=4/4" if ok else "multihop regression contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(regression)], "returncode": 1,
                       "output_tail": repr(exc)})
    binding = RUNS / "mixed-multihop-reader-registry-001.json"
    try:
        data = json.loads(binding.read_text(encoding="utf-8"))
        ok = bool(data.get("passed") and data.get("manifest_generation_bound")
                  and data.get("revoked_activation_blocked")
                  and data.get("revoked_manifest_blocked"))
        checks.append({"command": ["persisted", str(binding)], "returncode": 0 if ok else 1,
                       "output_tail": "generation-bound and revocation controls passed" if ok else "binding contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(binding)], "returncode": 1,
                       "output_tail": repr(exc)})
    full_eval = RUNS / "reader-lora-mixed-multihop-001" / "full-eval-report.json"
    try:
        data = json.loads(full_eval.read_text(encoding="utf-8"))
        ok = (data.get("two-hop-test", {}).get("correct") == 84
              and data.get("two-hop-test", {}).get("total") == 96
              and data.get("three-hop-test", {}).get("correct") == 4)
        checks.append({"command": ["persisted", str(full_eval)], "returncode": 0 if ok else 1,
                       "output_tail": "full split=84/96, three-hop=4/4" if ok else "full evaluation contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(full_eval)], "returncode": 1,
                       "output_tail": repr(exc)})
    boundary_eval = RUNS / "reader-lora-boundary-002" / "full-eval-report.json"
    try:
        data = json.loads(boundary_eval.read_text(encoding="utf-8"))
        ok = (data.get("two-hop-test", {}).get("correct") == 86
              and data.get("two-hop-test", {}).get("total") == 96
              and data.get("three-hop-test", {}).get("correct") == 4)
        checks.append({"command": ["persisted", str(boundary_eval)], "returncode": 0 if ok else 1,
                       "output_tail": "boundary candidate=86/96, three-hop=4/4" if ok else "boundary evaluation contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(boundary_eval)], "returncode": 1,
                       "output_tail": repr(exc)})
    boundary_binding = RUNS / "boundary-reader-registry-001.json"
    try:
        data = json.loads(boundary_binding.read_text(encoding="utf-8"))
        ok = bool(data.get("passed") and data.get("manifest_generation_bound")
                  and data.get("revoked_activation_blocked") and data.get("revoked_manifest_blocked"))
        checks.append({"command": ["persisted", str(boundary_binding)], "returncode": 0 if ok else 1,
                       "output_tail": "boundary candidate generation/revocation passed" if ok else "boundary binding contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(boundary_binding)], "returncode": 1,
                       "output_tail": repr(exc)})
    boundary_guard = RUNS / "reader-lora-boundary-002" / "guard-audit.json"
    try:
        data = json.loads(boundary_guard.read_text(encoding="utf-8"))
        ok = bool(data.get("passed") and data.get("two-hop-test", {}).get("guarded_exact") == 96
                  and data.get("three-hop-test", {}).get("guarded_exact") == 4)
        checks.append({"command": ["persisted", str(boundary_guard)], "returncode": 0 if ok else 1,
                       "output_tail": "typed guard=100/100" if ok else "typed guard contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(boundary_guard)], "returncode": 1,
                       "output_tail": repr(exc)})
    native_eval = RUNS / "native-embedding-reindex-001" / "report.json"
    try:
        data = json.loads(native_eval.read_text(encoding="utf-8"))
        ok = (data.get("recall_at_1") == 1.0 and data.get("dimension", 0) > 0
              and data.get("namespace", {}).get("schema", {}).get("text", {}).get("embed"))
        checks.append({"command": ["persisted", str(native_eval)], "returncode": 0 if ok else 1,
                       "output_tail": "native embedding recall@1=1.0" if ok else "native embedding contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(native_eval)], "returncode": 1,
                       "output_tail": repr(exc)})
    assembled_demo = RUNS / "assembled-stack-demo-001.json"
    try:
        data = json.loads(assembled_demo.read_text(encoding="utf-8"))
        ok = (data.get("schema") == "assembled-stack-demo:v1"
              and data.get("pods") == 3
              and all(row.get("route_matches_type") for row in data.get("queries", []))
              and data.get("revocation_removed_lead_time") is True)
        checks.append({"command": ["persisted", str(assembled_demo)], "returncode": 0 if ok else 1,
                       "output_tail": "assembled routing/retrieval/revocation passed" if ok else "assembled demo contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(assembled_demo)], "returncode": 1,
                       "output_tail": repr(exc)})
    gpu_throughput = RUNS / "reader-lora-boundary-002" / "gpu-throughput-report.json"
    try:
        data = json.loads(gpu_throughput.read_text(encoding="utf-8"))
        ok = (data.get("device") and data.get("tokens_per_s", 0) > 0
              and data.get("sequences_per_s", 0) > 0)
        checks.append({"command": ["persisted", str(gpu_throughput)], "returncode": 0 if ok else 1,
                       "output_tail": f"GPU generation={data.get('tokens_per_s', 0):.2f} tok/s" if ok else "GPU throughput contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(gpu_throughput)], "returncode": 1,
                       "output_tail": repr(exc)})
    pod_vs_rag = RUNS / "reader-lora-boundary-002" / "pod-vs-rag-throughput.json"
    try:
        data = json.loads(pod_vs_rag.read_text(encoding="utf-8"))
        pod = data.get("pod", {}); rag = data.get("rag", {})
        ok = (pod.get("batch_latency_ms", 0) > 0 and rag.get("batch_latency_ms", 0) > pod.get("batch_latency_ms", 0)
              and pod.get("input_tokens", 0) < rag.get("input_tokens", 0))
        checks.append({"command": ["persisted", str(pod_vs_rag)], "returncode": 0 if ok else 1,
                       "output_tail": f"Pod/RAG latency ratio={rag.get('batch_latency_ms', 0)/pod.get('batch_latency_ms', 1):.2f}x" if ok else "Pod/RAG timing contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(pod_vs_rag)], "returncode": 1,
                       "output_tail": repr(exc)})
    value_pod_vs_rag = RUNS / "multifact-internal-lora-004" / "pod-vs-rag-throughput.json"
    try:
        data = json.loads(value_pod_vs_rag.read_text(encoding="utf-8"))
        pod = data.get("pod", {}); rag = data.get("rag", {})
        ratio = rag.get("batch_latency_ms", 0) / max(pod.get("batch_latency_ms", 1), 1e-9)
        exact = pod.get("sample_output", "").rstrip().endswith("18 days") and rag.get("sample_output", "").rstrip().endswith("18 days")
        ok = ratio > 1.5 and pod.get("input_tokens", 0) < rag.get("input_tokens", 0) and exact
        checks.append({"command": ["persisted", str(value_pod_vs_rag)], "returncode": 0 if ok else 1,
                       "output_tail": f"value-bearing Pod/RAG={ratio:.2f}x, exact=both" if ok else "value-bearing Pod/RAG contract failed"})
    except Exception as exc:
        checks.append({"command": ["persisted", str(value_pod_vs_rag)], "returncode": 1,
                       "output_tail": repr(exc)})
    for script in ("r211b_stable_split_abi.py", "r212_intertwiner_calibration.py",
                   "r213_noisy_abi_calibration.py", "r214_semantic_root_multimodel.py"):
        checks.append(run([py, script], cwd=VM))

    report = {
        "schema": "neural-pods-project-gate:v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(c["returncode"] == 0 for c in checks),
        "checks": checks,
        "claims": {
            "full_research_goal_complete": False,
            "limits": [
                "private original So/CQP1/J-Space weights are unavailable",
                "R211-R214 are proxy/mechanism evidence",
                "broad external-world multi-hop quality is not established",
            ],
        },
    }
    out = RUNS / "project-gate-001.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(checks),
                      "report": str(out.relative_to(ROOT))}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
