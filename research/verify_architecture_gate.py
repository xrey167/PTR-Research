"""Fail-closed checks for the measured architecture evidence."""
from __future__ import annotations
import json
from pathlib import Path


def verify(path: str | Path = Path(__file__).with_name("runs") / "architecture-20260917.json") -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    vllm_path = Path(path).with_name("neohorse-vllm-20260917.jsonl")
    vllm_rows = [json.loads(line) for line in vllm_path.read_text(encoding="utf-8").splitlines() if line.strip()] if vllm_path.exists() else []
    vllm_peak = max((row.get("tokens_per_s", 0.0) for row in vllm_rows), default=0.0)
    vllm_ok = bool(vllm_rows) and all(row.get("errors") == 0 and row.get("ok", 0) == row.get("count") for row in vllm_rows) and vllm_peak >= 200.0
    dual_path = Path(path).with_name("neohorse-vllm-dual-20260917.jsonl")
    dual_rows = [json.loads(line) for line in dual_path.read_text(encoding="utf-8").splitlines() if line.strip()] if dual_path.exists() else []
    dual_peak = max((row.get("tokens_per_s", 0.0) for row in dual_rows), default=0.0)
    dual_ok = bool(dual_rows) and all(row.get("errors") == 0 and row.get("ok", 0) == row.get("count") for row in dual_rows) and dual_peak >= 350.0
    router_path = Path(path).with_name("vllm-router-20260917.jsonl")
    router_rows = [json.loads(line) for line in router_path.read_text(encoding="utf-8").splitlines() if line.strip()] if router_path.exists() else []
    router_ok = len(router_rows) >= 2 and router_rows[0].get("ok") == 128 and router_rows[0].get("errors") == 0 and router_rows[0].get("tokens_per_s", 0) >= 350 and router_rows[1].get("ok") == 32 and router_rows[1].get("errors") == 0 and router_rows[1].get("failovers") == 16
    lan_path = Path(path).with_name("vllm-router-lan-20260917.json")
    lan = json.loads(lan_path.read_text(encoding="utf-8")) if lan_path.exists() else {}
    lan_ok = lan.get("ok") == 128 and lan.get("errors") == 0 and lan.get("tokens_per_s", 0) >= 300 and lan.get("per_replica") == {"gpu0": 64, "gpu1": 64}
    lan_failover_path = Path(path).with_name("vllm-router-lan-failover-20260917.json")
    lan_failover = json.loads(lan_failover_path.read_text(encoding="utf-8")) if lan_failover_path.exists() else {}
    lan_failover_ok = lan_failover.get("ok") == 32 and lan_failover.get("errors") == 0 and lan_failover.get("failovers") == 16 and lan_failover.get("health") == {"gpu0": False, "gpu1": True}
    batch_path = Path(path).with_name("adaptive-batcher-20260917.json")
    batch = json.loads(batch_path.read_text(encoding="utf-8")) if batch_path.exists() else {}
    postgres_path = Path(path).with_name("postgres-quorum-20260917.json")
    postgres = json.loads(postgres_path.read_text(encoding="utf-8")) if postgres_path.exists() else {}
    multihost_path = Path(path).with_name("raft-multihost-20260917.json")
    multihost = json.loads(multihost_path.read_text(encoding="utf-8")) if multihost_path.exists() else {}
    pg_multihost_path = Path(path).with_name("postgres-quorum-multihost-20260919.json")
    pg_multihost = json.loads(pg_multihost_path.read_text(encoding="utf-8")) if pg_multihost_path.exists() else {}
    ensemble_path = Path(path).with_name("ensemble-20260919.json")
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8")) if ensemble_path.exists() else {}
    ensemble_metrics = ensemble.get("metrics", {})
    project_root = Path(path).resolve().parents[2]
    gen6_path = project_root / "runs" / "qwen3b-eval-test-gen6-20260919-report.json"
    gen6 = json.loads(gen6_path.read_text(encoding="utf-8")) if gen6_path.exists() else {}
    gen6_dev_path = project_root / "runs" / "qwen3b-eval-dev-gen6-20260919-report.json"
    gen6_dev = json.loads(gen6_dev_path.read_text(encoding="utf-8")) if gen6_dev_path.exists() else {}
    hetero_path = Path(path).with_name("ensemble-hetero-20260919.json")
    hetero = json.loads(hetero_path.read_text(encoding="utf-8")) if hetero_path.exists() else {}
    redis_cache_path = Path(path).with_name("redis-cache-20260919.json")
    redis_cache = json.loads(redis_cache_path.read_text(encoding="utf-8")) if redis_cache_path.exists() else {}
    grpc_path = Path(path).with_name("grpc-vs-tcp-20260919.json")
    grpc_cmp = json.loads(grpc_path.read_text(encoding="utf-8")) if grpc_path.exists() else {}
    project_root = Path(path).resolve().parents[2]
    lora_adapter_path = project_root / "runs" / "qwen3b-eval-test-adapter-20260917-report.json"
    lora_base_path = project_root / "runs" / "qwen3b-eval-test-base-20260917-report.json"
    lora_adapter = json.loads(lora_adapter_path.read_text(encoding="utf-8")) if lora_adapter_path.exists() else {}
    lora_base = json.loads(lora_base_path.read_text(encoding="utf-8")) if lora_base_path.exists() else {}
    dev_adapter_path = project_root / "runs" / "qwen3b-eval-dev-adapter-20260917-report.json"
    dev_base_path = project_root / "runs" / "qwen3b-eval-dev-base-20260917-report.json"
    dev_adapter = json.loads(dev_adapter_path.read_text(encoding="utf-8")) if dev_adapter_path.exists() else {}
    dev_base = json.loads(dev_base_path.read_text(encoding="utf-8")) if dev_base_path.exists() else {}
    gen4_adapter_path = project_root / "runs" / "qwen3b-eval-test-gen4-adapter-20260917-report.json"
    gen4_dev_adapter_path = project_root / "runs" / "qwen3b-eval-dev-gen4-adapter-20260917-report.json"
    gen4_adapter = json.loads(gen4_adapter_path.read_text(encoding="utf-8")) if gen4_adapter_path.exists() else {}
    gen4_dev_adapter = json.loads(gen4_dev_adapter_path.read_text(encoding="utf-8")) if gen4_dev_adapter_path.exists() else {}
    gen5_adapter_path = project_root / "runs" / "qwen3b-eval-test-gen5-20260919-report.json"
    gen5_dev_adapter_path = project_root / "runs" / "qwen3b-eval-dev-gen5-20260919-report.json"
    gen5_adapter = json.loads(gen5_adapter_path.read_text(encoding="utf-8")) if gen5_adapter_path.exists() else {}
    gen5_dev_adapter = json.loads(gen5_dev_adapter_path.read_text(encoding="utf-8")) if gen5_dev_adapter_path.exists() else {}
    checks = {
        "tests": d["tests"]["passed"] >= 260,
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
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("architecture gate failed: " + ", ".join(failed))
    return {"ok": True, "checks": checks}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
