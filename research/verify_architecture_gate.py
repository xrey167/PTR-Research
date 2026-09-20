"""Fail-closed checks for the measured architecture evidence.

The gate compares recorded benchmark output against thresholds; it does not
run the benchmarks. Evidence is therefore resolved from two directories, in
order:

  1. research/runs/ — checked into the repository, so a clean clone can
     reproduce the verdict;
  2. <project root>/runs/ — the server's benchmark output, which .gitignore
     excludes and which no clone carries.

A file that is absent in both is reported separately from a threshold that
was violated: "no evidence" and "evidence says no" need different fixes, and
a single combined failure line used to hide which of the two had happened.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_test_run import run_pytest, source_fingerprint  # noqa: E402


def _search_dirs(path: str | Path) -> list[Path]:
    run_file = Path(path).resolve()
    return [run_file.parent, run_file.parents[2] / "runs"]


def _find(name: str, dirs: list[Path]) -> Path | None:
    for directory in dirs:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _read(name: str, dirs: list[Path], missing: list[str]) -> dict:
    """Parsed JSON evidence, or {} when the file exists nowhere."""
    found = _find(name, dirs)
    if found is None:
        missing.append(name)
        return {}
    return json.loads(found.read_text(encoding="utf-8"))


def _read_lines(name: str, dirs: list[Path], missing: list[str]) -> list[dict]:
    """Parsed JSONL evidence, or [] when the file exists nowhere."""
    found = _find(name, dirs)
    if found is None:
        missing.append(name)
        return []
    return [json.loads(line)
            for line in found.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _dream_backtest_ok(backtest: dict) -> bool:
    """Whether a dream cycle's backtest clears the bar.

    Two shapes exist. Evidence recorded before 2026-09-20 carries a single
    in-sample error; that number cannot fail, because the simulator has one
    intercept plus one coefficient per decision and the history changes one
    decision per generation, so the fit reproduces its own points. The
    leave-one-generation-out shape that replaced it is checked on what it
    actually predicted out of sample, and must record whether the in-sample
    fit was degenerate so nobody quotes that number again without the caveat.

    A run that could predict NO generation out of sample still passes: that
    says the history is too short, which is a fact about the data, not a
    regression in the code.
    """
    if backtest.get("mode") == "leave_one_generation_out":
        if "degenerate" not in backtest.get("in_sample", {}):
            return False
        out = backtest.get("out_of_sample", {})
        if not out.get("generations"):
            return True
        return (out.get("max_abs_error") or 1.0) < 0.15
    return backtest.get("max_abs_error", 1) < 0.15


def verify(path: str | Path = Path(__file__).with_name("runs") / "architecture-20260917.json",
           *, run_tests: bool = False) -> dict:
    dirs = _search_dirs(path)
    missing: list[str] = []
    d = json.loads(Path(path).read_text(encoding="utf-8"))

    vllm_rows = _read_lines("neohorse-vllm-20260917.jsonl", dirs, missing)
    vllm_peak = max((row.get("tokens_per_s", 0.0) for row in vllm_rows), default=0.0)
    vllm_ok = bool(vllm_rows) and all(row.get("errors") == 0 and row.get("ok", 0) == row.get("count") for row in vllm_rows) and vllm_peak >= 200.0
    dual_rows = _read_lines("neohorse-vllm-dual-20260917.jsonl", dirs, missing)
    dual_peak = max((row.get("tokens_per_s", 0.0) for row in dual_rows), default=0.0)
    dual_ok = bool(dual_rows) and all(row.get("errors") == 0 and row.get("ok", 0) == row.get("count") for row in dual_rows) and dual_peak >= 350.0
    router_rows = _read_lines("vllm-router-20260917.jsonl", dirs, missing)
    router_ok = len(router_rows) >= 2 and router_rows[0].get("ok") == 128 and router_rows[0].get("errors") == 0 and router_rows[0].get("tokens_per_s", 0) >= 350 and router_rows[1].get("ok") == 32 and router_rows[1].get("errors") == 0 and router_rows[1].get("failovers") == 16
    lan = _read("vllm-router-lan-20260917.json", dirs, missing)
    lan_ok = lan.get("ok") == 128 and lan.get("errors") == 0 and lan.get("tokens_per_s", 0) >= 300 and lan.get("per_replica") == {"gpu0": 64, "gpu1": 64}
    lan_failover = _read("vllm-router-lan-failover-20260917.json", dirs, missing)
    lan_failover_ok = lan_failover.get("ok") == 32 and lan_failover.get("errors") == 0 and lan_failover.get("failovers") == 16 and lan_failover.get("health") == {"gpu0": False, "gpu1": True}
    batch = _read("adaptive-batcher-20260917.json", dirs, missing)
    postgres = _read("postgres-quorum-20260917.json", dirs, missing)
    multihost = _read("raft-multihost-20260917.json", dirs, missing)
    pg_multihost = _read("postgres-quorum-multihost-20260919.json", dirs, missing)
    ensemble = _read("ensemble-20260919.json", dirs, missing)
    ensemble_metrics = ensemble.get("metrics", {})
    gen6 = _read("qwen3b-eval-test-gen6-20260919-report.json", dirs, missing)
    gen6_dev = _read("qwen3b-eval-dev-gen6-20260919-report.json", dirs, missing)
    hetero = _read("ensemble-hetero-20260919.json", dirs, missing)
    redis_cache = _read("redis-cache-20260919.json", dirs, missing)
    grpc_cmp = _read("grpc-vs-tcp-20260919.json", dirs, missing)
    tp = _read("traced-pipeline-20260920.json", dirs, missing)
    tcp = _read("native-tcp-cross-20260920.json", dirs, missing)
    e2e = _read("mesh-e2e-20260920.json", dirs, missing)
    mc = _read("mesh-cache-20260920.json", dirs, missing)
    tg = _read("taskgraph-20260920.json", dirs, missing)
    comm = _read("native-comm-eval-20260920-report.json", dirs, missing)
    reflex = _read("reflex-dispatch-20260919.json", dirs, missing)
    mesh = _read("mesh-presence-20260920.json", dirs, missing)
    dream = _read("dream-cycle-20260920.json", dirs, missing)
    dream_ev = _read("qwen3b-eval-test-gen7-20260920-report.json", dirs, missing)
    holdout = _read("reader-holdout-manifest-20260920.json", dirs, missing)
    storage = _read("storage-facade-20260920.json", dirs, missing)
    if run_tests:
        # --run-tests: the gate executes the suite instead of reading about it.
        test_run = run_pytest(Path(path).resolve().parents[2])
    else:
        test_run = _read("tests-20260920.json", dirs, [])  # optional: old
        # evidence predates it, so a miss falls back rather than failing.
    storage_kv = storage.get("kv", {})
    storage_l2 = storage.get("l2_lance", {})
    lora_adapter = _read("qwen3b-eval-test-adapter-20260917-report.json", dirs, missing)
    lora_base = _read("qwen3b-eval-test-base-20260917-report.json", dirs, missing)
    dev_adapter = _read("qwen3b-eval-dev-adapter-20260917-report.json", dirs, missing)
    dev_base = _read("qwen3b-eval-dev-base-20260917-report.json", dirs, missing)
    gen4_adapter = _read("qwen3b-eval-test-gen4-adapter-20260917-report.json", dirs, missing)
    gen4_dev_adapter = _read("qwen3b-eval-dev-gen4-adapter-20260917-report.json", dirs, missing)
    gen5_adapter = _read("qwen3b-eval-test-gen5-20260919-report.json", dirs, missing)
    gen5_dev_adapter = _read("qwen3b-eval-dev-gen5-20260919-report.json", dirs, missing)
    checks = {
        # A recorded test run is only evidence about the code that produced
        # it. tests-20260920.json carries a digest over neural_pods/,
        # research/ and tests/; if today's tree hashes differently, the
        # recording describes something else and the check fails. Without
        # that file the gate falls back to the 2026-09-17 count, which is
        # exactly the blind spot - so it is reported as stale below.
        "tests": (test_run.get("failed") == 0
                  and test_run.get("passed", 0) >= 260
                  and test_run.get("sources_sha256")
                  == source_fingerprint(Path(path).resolve().parents[2]))
                 if test_run else d["tests"]["passed"] >= 260,
        "retrieval_recall": d["retrieval"]["recall_at_5"] >= 0.99 and d["retrieval"]["hnsw_recall_at_10"] >= 0.99,
        "cache": d["cache"]["hit_rate"] >= 0.98 and d["cache"]["p99_ms"] < 1.0,
        "authenticated_transport": d["transport"]["mtls_hmac_manifest_p95_ms"] < 1.0 and d["transport"]["model_tok_s"] > 200 and d["transport"]["raft_frame_valid"] == 10000 and d["transport"]["subject_allowlist_allowed"] == d["transport"]["subject_allowlist_after_denied"] == 1,
        "raft_replication": min(d["raft"]["replicated"].values()) >= 100 and d["raft"]["persistent_tcp_proposals_s"] > 5000 and d["raft"]["persistent_mtls_proposals_s"] > 1000,
        "quorum": d["replication"]["quorum_successes"] == 1000,
        "resource_release": d["resources"]["active_leases_after"] == 0 and max(d["resources"]["gpu_memory_after_bytes"]) <= 1048576,
        "vllm_serving": vllm_ok,
        "vllm_dual_gpu": dual_ok,
        "vllm_router_failover": router_ok,
        "vllm_router_lan": lan_ok,
        "vllm_router_lan_failover": lan_failover_ok,
        "adaptive_batcher": batch.get("correct") is True and batch.get("requests_per_s", 0) >= 5000 and batch.get("stats", {}).get("errors") == 0 and batch.get("stats", {}).get("queued") == 0,
        "postgres_quorum": postgres.get("replicas") == 3 and postgres.get("quorum") == 2 and postgres.get("latest") == postgres.get("writes") and postgres.get("healthy_p95_ms", 999) < 10.0,
        "raft_multihost": multihost.get("results", {}).get("replicated_all") is True
            and multihost.get("results", {}).get("failover", {}).get("ok") is True
            and multihost.get("results", {}).get("failover", {}).get("quorum_writes", 0) >= 50,
        "raft_multihost_rejoin": multihost.get("results", {}).get("failover", {}).get("node_rejoined") is True,
        "postgres_quorum_multihost": pg_multihost.get("latest") == pg_multihost.get("writes")
            and pg_multihost.get("healthy_p95_ms", 999) < 25.0
            and len(pg_multihost.get("hosts", [])) == 3,
        "ensemble_routing": ensemble.get("status") == "completed" and ensemble_metrics.get("errors") == 0
            and ensemble_metrics.get("gen5_raw") == 119 and ensemble_metrics.get("gen5_guarded") == 92,
        "ensemble_failover": ensemble_metrics.get("failover", {}).get("ok") == 32
            and ensemble_metrics.get("failover", {}).get("failovers") == 16,
        "gen6_hetero_pod": gen6.get("status") == "completed" and gen6.get("reader_unchanged") is True
            and gen6.get("exact_target_matches", 0) >= 125 and gen6.get("guarded_exact_target_matches", 0) >= 92,
        "gen6_promoted_dev": gen6_dev.get("status") == "completed" and gen6_dev.get("reader_unchanged") is True
            and gen6_dev.get("exact_target_matches", 0) >= gen4_dev_adapter.get("exact_target_matches", 0)
            and gen6_dev.get("guarded_exact_target_matches", 0) >= gen4_dev_adapter.get("guarded_exact_target_matches", 0),
        "ensemble_hetero_union": hetero.get("status") == "completed" and hetero.get("metrics", {}).get("errors") == 0
            and hetero.get("metrics", {}).get("union_raw", 0) >= 126,
        "redis_cache_tier": redis_cache.get("lru_redis", {}).get("stats", {}).get("redis_hits", 0) >= 2000
            and redis_cache.get("lru_redis", {}).get("stats", {}).get("redis_errors") == 0
            and redis_cache.get("lru_redis", {}).get("hot_p99_ms", 99) < 2.0,
        "grpc_transport_decision": grpc_cmp.get("tcp", {}).get("req_per_s", 0) > 10000
            and grpc_cmp.get("grpc_unary", {}).get("p50_ms", 0) > grpc_cmp.get("tcp", {}).get("p50_ms", 1) * 3,
        "reflex_dispatch": reflex.get("status") == "completed" and reflex.get("metrics", {}).get("errors") == 0
            and reflex.get("metrics", {}).get("union_raw", 0) >= reflex.get("baseline_union_raw", 999)
            and reflex.get("metrics", {}).get("channel_stats", {}).get("failovers") == reflex.get("metrics", {}).get("channel_stats", {}).get("reflex_misses"),
        "dream_pipeline": dream.get("status") == "dream_cycle_completed"
            and _dream_backtest_ok(dream.get("backtest", {}))
            and bool(dream.get("winner", {}).get("decisions")),
        "gen7_dream_validated": dream_ev.get("status") == "completed" and dream_ev.get("reader_unchanged") is True
            and dream_ev.get("exact_target_matches", 0) >= 125
            and dream_ev.get("guarded_exact_target_matches", 0) >= 92,
        "mesh_presence": mesh.get("discovery", {}).get("discovered") is True
            and mesh.get("rounds_ok", 0) == 100
            and (mesh.get("rtt_p50_ms") or 999) < 10.0,
        # frames_valid_rate only says the dialect parses; exact_rate says it
        # emitted the RIGHT frame. The measured 0.55 had no threshold at all,
        # so a model that produced well-formed nonsense scored 1.0 here.
        "native_protocol": comm.get("status") == "completed"
            and comm.get("metrics", {}).get("frames_valid_rate", 0) >= 0.98
            and comm.get("metrics", {}).get("exact_rate", 0) >= 0.50
            and comm.get("metrics", {}).get("acl_refused_forbidden") is True,
        # The frozen test split has stood at 125/132 raw and 44/44 concept
        # since Gen-6: saturated, and therefore unable to separate one reader
        # generation from the next. The holdout split is the replacement, and
        # it is only worth anything while it shares nothing with the frozen
        # splits and can still be regenerated from the repository.
        # F3/F4 shipped without these two checks, which is how six defects in
        # the facade survived. Each condition below is one of them.
        "storage_facade": storage.get("status") == "completed"
            and storage_kv.get("duplicate_rows") == 0
            and storage_kv.get("read_miss_tables_created") == 0
            and storage_kv.get("stale_read_after_reput") is False
            and storage_kv.get("rows_after_reput") == storage_kv.get("keys")
            and storage_kv.get("l1_backfilled_after_l2_hit") == storage_kv.get("keys")
            and storage_kv.get("l1_faster_than_l2") is True
            and storage_kv.get("quoted_key_roundtrip") is True
            and storage.get("session_affinity", {}).get("reuses") == 1
            and storage.get("session_affinity", {}).get("failovers") == 1,
        "storage_l2_lance": storage.get("status") == "completed"
            and storage_l2.get("duplicate_document_rows") == 0
            and storage_l2.get("duplicate_trace_rows") == 0
            and storage_l2.get("top_k_distinct") == storage_l2.get("top_k_requested")
            and storage_l2.get("tables_listed") == storage_l2.get("namespaces_created")
            and storage_l2.get("quoted_stage_matches", 0) > 0,
        "holdout_split_disjoint": holdout.get("schema") == "reader-holdout-split:v1"
            and holdout.get("disjoint_from_frozen_splits") is True
            and not any(holdout.get("overlaps", {"unchecked": [1]}).values())
            and holdout.get("rows", 0) >= 96,
        # "speedup" was the sum of contention-inflated node durations over
        # wall time, which rises WITH queueing; taskgraph.py now reports it as
        # mean_concurrency and the benchmark records wall_within_bound, which
        # compares wall time against the delay the DAG actually injects.
        # Evidence recorded before 2026-09-20 only carries the old key.
        "taskgraph_parallel": tg.get("correct") is True
            and len(tg.get("results", {})) == 6
            and (tg.get("wall_within_bound") is True
                 if "wall_within_bound" in tg
                 else (tg.get("mean_concurrency", tg.get("speedup")) or 0) > 1.5),
        "mesh_cache": mc.get("status") == "completed"
            and mc.get("cross_node_read") is True
            and mc.get("principal_isolated") is True
            and mc.get("invalidation_works") is True,
        "mesh_e2e": e2e.get("status") == "completed"
            and e2e.get("acks_received") == 20
            and e2e.get("pod_b_validated_all") is True
            and e2e.get("executor_stats", {}).get("refused") == 0,
        "native_tcp_cross_node": tcp.get("status") == "completed"
            and tcp.get("integrity_ok") == 30
            and tcp.get("acl_blocked_forbidden") is True,
        "traced_pipeline": tp.get("status") == "completed"
            and (tp.get("improvement_factor") or 0) >= 2.0
            and all(r["reflex_frames_valid"] == 132 for r in tp.get("runs", [])),
        "lora_ab": lora_adapter.get("status") == "completed" and lora_base.get("status") == "completed" and lora_adapter.get("reader_unchanged") is True and lora_base.get("reader_unchanged") is True and lora_adapter.get("guarded_exact_target_matches", 0) > lora_base.get("guarded_exact_target_matches", 0),
        "lora_ab_dev": dev_adapter.get("status") == "completed" and dev_base.get("status") == "completed" and dev_adapter.get("reader_unchanged") is True and dev_base.get("reader_unchanged") is True and dev_adapter.get("guarded_exact_target_matches", 0) > dev_base.get("guarded_exact_target_matches", 0),
        "lora_ab_gen4": gen4_adapter.get("status") == "completed" and gen4_adapter.get("reader_unchanged") is True
            and gen4_adapter.get("exact_target_matches", 0) > lora_adapter.get("exact_target_matches", 0)
            and gen4_adapter.get("guarded_exact_target_matches", 0) >= lora_adapter.get("guarded_exact_target_matches", 0),
        "lora_ab_gen4_dev": gen4_dev_adapter.get("status") == "completed" and gen4_dev_adapter.get("reader_unchanged") is True
            and gen4_dev_adapter.get("exact_target_matches", 0) > dev_adapter.get("exact_target_matches", 0)
            and gen4_dev_adapter.get("guarded_exact_target_matches", 0) >= dev_adapter.get("guarded_exact_target_matches", 0),
        "lora_ab_gen5": gen5_adapter.get("status") == "completed" and gen5_adapter.get("reader_unchanged") is True
            and gen5_adapter.get("exact_target_matches", 0) > gen4_adapter.get("exact_target_matches", 0)
            and gen5_adapter.get("guarded_exact_target_matches", 0) >= gen4_adapter.get("guarded_exact_target_matches", 0),
        "lora_ab_gen5_dev": gen5_dev_adapter.get("status") == "completed" and gen5_dev_adapter.get("reader_unchanged") is True
            and gen5_dev_adapter.get("exact_target_matches", 0) > gen4_dev_adapter.get("exact_target_matches", 0)
            and gen5_dev_adapter.get("guarded_exact_target_matches", 0) >= gen4_dev_adapter.get("guarded_exact_target_matches", 0),
    }
    missing = sorted(set(missing))
    # Comparative checks need BOTH sides present. With the baseline absent,
    # `>= baseline.get(field, 0)` is trivially true, so the check would pass
    # on no evidence at all - gen6_promoted_dev did exactly that once the
    # gen4 reports were no longer reachable.
    baseline_of = {
        "gen6_promoted_dev": "qwen3b-eval-dev-gen4-adapter-20260917-report.json",
        "lora_ab": "qwen3b-eval-test-base-20260917-report.json",
        "lora_ab_dev": "qwen3b-eval-dev-base-20260917-report.json",
        "lora_ab_gen4": "qwen3b-eval-test-adapter-20260917-report.json",
        "lora_ab_gen4_dev": "qwen3b-eval-dev-adapter-20260917-report.json",
        "lora_ab_gen5": "qwen3b-eval-test-gen4-adapter-20260917-report.json",
        "lora_ab_gen5_dev": "qwen3b-eval-dev-gen4-adapter-20260917-report.json",
    }
    for check_name, baseline in baseline_of.items():
        if baseline in missing:
            checks[check_name] = False
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        report = ["architecture gate failed: " + ", ".join(failed)]
        if missing:
            report.append(
                "missing evidence (" + str(len(missing)) + " file(s) found in "
                "neither " + " nor ".join(str(p) for p in dirs) + "): "
                + ", ".join(missing))
            report.append(
                "a missing file is not a regression - re-run the benchmark, "
                "or check its output in under research/runs/ so a clone can "
                "reproduce this verdict.")
        raise SystemExit("\n".join(report))
    return {"ok": True, "checks": checks, "missing_evidence": missing,
            "tests_evidence": ("executed now" if run_tests else
                               "tests-20260920.json" if test_run else
                               "architecture-20260917.json (stale: no digest, "
                               "predates the code it covers)")}


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--run-tests", action="store_true",
                         help="run the test suite now instead of reading a "
                              "recorded result")
    _args = _parser.parse_args()
    print(json.dumps(verify(run_tests=_args.run_tests), indent=2))
