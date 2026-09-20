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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods import architecture  # noqa: E402
from evidence import script_sha256, subject_sha256  # noqa: E402
from record_test_run import run_pytest, source_fingerprint  # noqa: E402


RESEARCH = Path(__file__).resolve().parent

#: Evidence files loaded during the current verify() call, name -> parsed
#: JSON. Filled by _read() so the staleness pass below covers EVERY file the
#: gate reads instead of a hand-kept tuple that fell 30 files behind. The one
#: list a person still maintains is the grandfather set.
_LOADED: dict[str, dict] = {}

#: Evidence recorded before research/evidence.py existed (2026-09-20) and not
#: reproducible without the server. Unstamped evidence normally FAILS the
#: gate — a file whose producer cannot be identified is not evidence — and
#: this set is the ratchet that lets the existing debt be paid off without
#: the gate being permanently red in the meantime. It may only ever shrink:
#: tests/test_architecture_gate.py fails if a name in here is now stamped, or
#: if a file outside it is unstamped. Remove a name by re-running its
#: benchmark through research.evidence.write().
UNSTAMPED_GRANDFATHERED = frozenset({
    "adaptive-batcher-20260917.json",
    "dream-cycle-20260920.json",
    "ensemble-20260919.json",
    "ensemble-hetero-20260919.json",
    "grpc-vs-tcp-20260919.json",
    "mesh-cache-20260920.json",
    "mesh-e2e-20260920.json",
    "mesh-presence-20260920.json",
    "native-comm-eval-20260920-report.json",
    "native-tcp-cross-20260920.json",
    "postgres-quorum-20260917.json",
    "postgres-quorum-multihost-20260919.json",
    "qwen3b-eval-dev-adapter-20260917-report.json",
    "qwen3b-eval-dev-base-20260917-report.json",
    "qwen3b-eval-dev-gen4-adapter-20260917-report.json",
    "qwen3b-eval-dev-gen5-20260919-report.json",
    "qwen3b-eval-dev-gen6-20260919-report.json",
    "qwen3b-eval-test-adapter-20260917-report.json",
    "qwen3b-eval-test-base-20260917-report.json",
    "qwen3b-eval-test-gen4-adapter-20260917-report.json",
    "qwen3b-eval-test-gen5-20260919-report.json",
    "qwen3b-eval-test-gen6-20260919-report.json",
    "qwen3b-eval-test-gen7-20260920-report.json",
    "raft-multihost-20260917.json",
    "reader-holdout-manifest-20260920.json",
    "redis-cache-20260919.json",
    "reflex-dispatch-20260919.json",
    "taskgraph-20260920.json",
    "tests-20260920.json",
    "traced-pipeline-20260920.json",
    "vllm-router-lan-20260917.json",
    "vllm-router-lan-failover-20260917.json",
})


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
    """Parsed JSON evidence, or {} when the file exists nowhere.

    Every file read here is registered in `_LOADED`, which is what the
    producer-staleness pass iterates. Registering at the point of reading is
    the whole point: a new check cannot acquire evidence that escapes the
    staleness pass by someone forgetting to add it to a second list.
    """
    found = _find(name, dirs)
    if found is None:
        missing.append(name)
        _LOADED[name] = {}
        return {}
    data = json.loads(found.read_text(encoding="utf-8"))
    _LOADED[name] = data
    return data


def _producer_status(name: str, data: dict) -> tuple[str | None, str | None]:
    """(stale, unstamped) for one evidence file.

    TWO drifts, both fail-closed.

    The producer drift: a benchmark is edited while its recorded output stays
    behind, and research/runs/ holds numbers the repository can no longer
    produce — the defect the 2026-09-20 review found in the frozen evaluation
    splits and which this gate then reproduced for taskgraph and
    traced-pipeline.

    The subject drift, which is the wider one: the SYSTEM changes while the
    benchmark does not. Before `subject_sha256`, gutting any of seven core
    modules turned not one check red, because no evidence was bound to the
    code it was evidence about. A file that names a subject is now checked
    against that subject's current bytes.
    """
    producer = data.get("producer")
    if not data:
        return None, None
    if not producer:
        return None, name
    script = RESEARCH / producer
    if not script.exists():
        return f"{name}: its producer {producer} no longer exists", None
    if data.get("producer_sha256") != script_sha256(script):
        return (f"{name}: {producer} has changed since this was recorded - "
                "re-run it"), None
    subject = data.get("subject")
    if subject and data.get("subject_sha256") != subject_sha256(subject):
        return (f"{name}: the code it measured has changed since this was "
                f"recorded ({', '.join(subject)}) - re-run {producer}"), None
    return None, None


def _read_lines(name: str, dirs: list[Path], missing: list[str]) -> list[dict]:
    """Parsed JSONL evidence, or [] when the file exists nowhere."""
    found = _find(name, dirs)
    if found is None:
        missing.append(name)
        return []
    return [json.loads(line)
            for line in found.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _taskgraph_ok(tg: dict, legacy: list[str]) -> bool:
    """Prefer wall_within_bound; fall back to the pre-2026-09-20 ratio.

    The fallback is not equivalent. wall_within_bound compares wall time
    against the delay the DAG injects and so distinguishes parallel from
    serial execution; the old ratio rises WITH queueing and cannot. Evidence
    that only carries the old key is recorded in `legacy` so a green check on
    it is not mistaken for the stricter one having passed.
    """
    if "wall_within_bound" in tg:
        return tg["wall_within_bound"] is True
    legacy.append("taskgraph_parallel: no wall_within_bound in the recorded "
                  "evidence, fell back to mean_concurrency/speedup > 1.5; "
                  "re-run research/benchmark_taskgraph.py")
    return (tg.get("mean_concurrency", tg.get("speedup")) or 0) > 1.5


def _dream_backtest_well_formed(backtest: dict) -> bool:
    """Whether a dream cycle's backtest was measured the way it must be.

    Two shapes exist. Evidence recorded before 2026-09-20 carries a single
    in-sample error; that number cannot fail, because the simulator has one
    intercept plus one coefficient per decision and the history changes one
    decision per generation, so the fit reproduces its own points. A check
    that passes on it passes on nothing, so this returns False for it: the
    cycle has to be re-run with the leave-one-generation-out simulator.

    The newer shape must record whether the in-sample fit was degenerate, so
    nobody quotes that number again without the caveat. This function stops
    there. Whether the run predicted anything is a separate question, asked
    by _dream_backtest_predictive and carried by its own gate check: a cycle
    that ran correctly over a history too short to predict from is a fact
    about the data, and it must not be confused with a cycle that ran.
    """
    if backtest.get("mode") != "leave_one_generation_out":
        return False
    return "degenerate" in backtest.get("in_sample", {})


def _dream_backtest_predictive(backtest: dict) -> bool:
    """Whether the replay simulator predicted a held-out generation at all.

    This is the claim "the dream pod can estimate an outcome it has not
    seen". It needs at least one generation predicted OUT of sample and an
    error under the same 0.15 bound the cycle aborts on. As of 2026-09-20 the
    recorded history identifies none: four generations, each changing one
    decision, leave the remaining three unable to pin the coefficients. The
    check stays red until the history is long enough — red because unproven,
    not because broken.
    """
    if not _dream_backtest_well_formed(backtest):
        return False
    out = backtest.get("out_of_sample", {})
    if not out.get("generations"):
        return False
    # A MISSING measurement fails; a measured one is compared. `or 1.0`
    # conflated the two, because 0.0 is falsy — so a run that predicted a
    # held-out generation exactly was the one result this check rejected.
    # Same rule as the mesh_presence check states two hundred lines down.
    max_abs_error = out.get("max_abs_error")
    if max_abs_error is None:
        return False
    return max_abs_error < 0.15


def verify(path: str | Path = Path(__file__).with_name("runs") / "architecture-20260917.json",
           *, run_tests: bool = False) -> dict:
    dirs = _search_dirs(path)
    _LOADED.clear()
    missing: list[str] = []
    # Evidence recorded before 2026-09-20 carries older shapes that several
    # checks still accept. Which ones took that path is reported rather than
    # left to the reader: an accepted legacy shape means the newer, stricter
    # criterion has not actually been applied yet.
    legacy: list[str] = []
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
    dream_reflex = _read("dream-reflex-20260920.json", dirs, missing)
    storage = _read("storage-facade-20260920.json", dirs, missing)
    xgboost_pod = _read("xgboost-pod-20260920.json", dirs, missing)
    stale_evidence: list[str] = []
    unstamped_evidence: list[str] = []

    # Static analysis, not recorded evidence: it costs milliseconds and it
    # describes the tree as it is right now, so there is nothing to record.
    layering = architecture.check()
    if run_tests:
        # --run-tests: the gate executes the suite instead of reading about it.
        test_run = run_pytest(Path(path).resolve().parents[2])
    else:
        # Reported as missing evidence like every other file. It used to be
        # read into a throwaway list so a clone without it fell back to a
        # count from 2026-09-17 - a green `tests` on a number recorded days
        # before the code, with no source fingerprint behind it.
        test_run = _read("tests-20260920.json", dirs, missing)
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
        # that file there is nothing to fall back to: an absent recording is
        # missing evidence, not a passed suite.
        # ARCHITECTURE-MASTER section 1 states the layer rule and the storage
        # facade rule in prose. This is the same statement, checked: a module
        # that imports upward, reaches a backend past the facade, is added
        # without being placed on a layer, or forms a runtime cycle fails it.
        "layering": layering["ok"],
        # The exit code comes FIRST. Everything else here is reconstructed
        # from pytest's summary line, and that line reports collection and
        # fixture failures as `errors`: "300 passed, 4 errors" used to parse
        # to failed 0 and pass. `errors == 0` closes that, and the skip
        # ceiling closes the other half — a suite that stops running a third
        # of itself must not stay green on its passed-count alone.
        "tests": (bool(test_run)
                  and test_run.get("exit_code") == 0
                  and test_run.get("failed") == 0
                  and test_run.get("errors", 0) == 0
                  and test_run.get("passed", 0) >= 260
                  and test_run.get("skipped", 0) <= 20
                  and test_run.get("sources_sha256")
                  == source_fingerprint(Path(path).resolve().parents[2])),
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
        # One check used to carry two claims, and the weaker one made the
        # stronger one look proven. Split, so each says what it says:
        #
        # reflex_failover — the RETRACT mechanism. Every miss was caught by
        # the default pod, quality did not drop against the static baseline,
        # no dispatch raised. This is what the 2026-09-19 run demonstrated,
        # and it is green.
        #
        # reflex_dispatch — the ADDRESS actually resolving. The same run
        # recorded reflex_hits 0, reflex_misses 132, failovers 132: not one
        # address signal resolved, every answer came from the default pod.
        # The old check was green on that evidence, because it looked at
        # neither the hit count nor at a real error counter (it read
        # metrics.errors, which the benchmark sets to 0 and never
        # increments). Red until the addressing this phase is named for
        # exists — POD-ARM-DESIGN P5, latent address training.
        "reflex_failover": reflex.get("status") == "completed"
            and reflex.get("metrics", {}).get("channel_stats", {}).get("errors") == 0
            and reflex.get("metrics", {}).get("union_raw", 0) >= reflex.get("baseline_union_raw", 999)
            and reflex.get("metrics", {}).get("channel_stats", {}).get("failovers") == reflex.get("metrics", {}).get("channel_stats", {}).get("reflex_misses"),
        "reflex_dispatch": reflex.get("status") == "completed"
            and reflex.get("metrics", {}).get("channel_stats", {}).get("reflex_hits", 0) > 0,
        # One check used to carry two claims again, the same way
        # reflex_failover/reflex_dispatch did:
        #
        # dream_pipeline — the cycle RAN. It completed, it produced a winner
        # the history can identify, and its backtest has the shape that can
        # fail (leave-one-generation-out, with the in-sample degeneracy
        # recorded). It says nothing about predictive power.
        #
        # dream_predictive — the cycle PREDICTED something it had not seen.
        # At least one held-out generation, error under the 0.15 bound the
        # cycle itself aborts on.
        #
        # Both are red on the evidence checked in today, because that file
        # carries the pre-2026-09-20 in-sample number, which cannot fail. The
        # cycle has to be re-run on the machine that holds the generation
        # reports; until then a green dream_pipeline would be green on
        # nothing. `estimable is not False` still accepts evidence without
        # that key; what it rules out is a winner the history cannot identify.
        "dream_pipeline": dream.get("status") == "dream_cycle_completed"
            and _dream_backtest_well_formed(dream.get("backtest", {}))
            and bool(dream.get("winner", {}).get("decisions"))
            and dream.get("winner", {}).get("estimable") is not False,
        "dream_predictive": _dream_backtest_predictive(dream.get("backtest", {})),
        "gen7_dream_validated": dream_ev.get("status") == "completed" and dream_ev.get("reader_unchanged") is True
            and dream_ev.get("exact_target_matches", 0) >= 125
            and dream_ev.get("guarded_exact_target_matches", 0) >= 92,
        # D3 of the Dream-Pod design counted as done with no check behind it.
        # This one covers the binding, not the policy: the alias resolves, the
        # dream pod answers deterministically, a miss retracts to the default
        # pod, and resolution stays under the Pod-Arm latency target.
        # pool_source is checked, not just recorded: the benchmark falls back
        # to a two-generation synthetic pool in a checkout without the
        # generation reports, and a latency measured over that says nothing
        # about the real one. The producer hash cannot catch it — same script,
        # different input.
        "dream_reflex": dream_reflex.get("status") == "completed"
            and str(dream_reflex.get("pool_source", "")).startswith("recorded")
            and dream_reflex.get("miss_retracted_to_default") is True
            and dream_reflex.get("winner_deterministic") is True
            and dream_reflex.get("resolve_within_target") is True
            and dream_reflex.get("reflex", {}).get("all_misses_covered") is True
            and dream_reflex.get("reflex", {}).get("errors") == 0,
        # P2 of the Pod-Arm design promised this check and shipped without
        # it, so what held the phase up was a unit test of the factory. Four
        # conditions, one per claim the phase makes: the runtime is reachable
        # through the factory by kind, a real ResourceGovernor refuses a pod
        # that does not fit, two independently activated pods answer
        # identically, and release gives every byte back.
        "xgboost_pod": xgboost_pod.get("status") == "completed"
            and xgboost_pod.get("created_via_factory") is True
            and xgboost_pod.get("budget_refused_third_pod") is True
            and xgboost_pod.get("peak_matches_two_pods") is True
            and xgboost_pod.get("deterministic_across_pods") is True
            and xgboost_pod.get("deterministic_across_calls") is True
            and xgboost_pod.get("released_to_zero") is True
            and xgboost_pod.get("reactivation_after_release") is True
            # A model that answers the same class for every probe would make
            # the determinism above vacuous.
            and xgboost_pod.get("distinct_predictions", 0) >= 2,
        "mesh_presence": mesh.get("discovery", {}).get("discovered") is True
            and mesh.get("rounds_ok", 0) == 100
            # `or 999` here would have turned a legitimate 0.0 into a
            # failure: only a MISSING measurement may fail this clause.
            and mesh.get("rtt_p50_ms") is not None
            and mesh["rtt_p50_ms"] < 10.0,
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
            # `quoted_key_roundtrip` used to be the literal True in the
            # report, so this clause asserted nothing. It is now derived
            # from a count of round-trips that returned what was written.
            and storage_kv.get("quoted_key_roundtrips_attempted", 0) > 0
            # A read must leave nothing behind AND return nothing. The
            # second half used to be an assert inside the benchmark, which
            # killed the run instead of recording the failure.
            and storage_kv.get("read_miss_returned_nothing") is True
            # A store that already held data before the l1_eligible column
            # existed. The benchmark starts from a fresh temp dir every run,
            # so this case was invisible to the gate until it was measured
            # explicitly — and lancedb refuses the unknown column outright,
            # which would have broken put() on every grown store.
            and storage_kv.get("legacy_table_migrated") is True
            and storage_kv.get("legacy_row_readable") is True
            and storage_kv.get("legacy_row_backfilled_l1", 0) >= 1
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
            and _taskgraph_ok(tg, legacy),
        # `principal_isolated` used to be the third condition. The benchmark
        # produced it by reading a principal nothing had ever been written
        # under, so it was true by construction — while the same script read
        # another principal's entry across the node boundary two lines
        # earlier. What is checkable is the client-side refusal; that the
        # entries themselves are reachable with the Redis credential is
        # recorded as a fact, not asserted away.
        "mesh_cache": mc.get("status") == "completed"
            and mc.get("cross_node_read") is True
            and mc.get("cross_principal_refused_by_client") is True
            and mc.get("isolation") == "client-side key derivation"
            and mc.get("invalidation_works") is True,
        "mesh_e2e": e2e.get("status") == "completed"
            and e2e.get("acks_received") == 20
            and e2e.get("pod_b_validated_all") is True
            and e2e.get("executor_stats", {}).get("refused") == 0,
        "native_tcp_cross_node": tcp.get("status") == "completed"
            and tcp.get("integrity_ok") == 30
            and tcp.get("acl_blocked_forbidden") is True,
        # `reflex_frames_valid` was neither about the reflex nor a test: the
        # benchmark builds the frame with an f-string over json.dumps and then
        # parses its own output, so the count equalled the case count by
        # construction. It is now recorded under the name of what it actually
        # checks — that the SERIALISER produces frames the parser accepts —
        # and read under that name here. Whether the MODEL can produce them is
        # what native_protocol measures (exact_rate 0.55), and nothing in this
        # check may be read as saying anything about that.
        # `runs_comparable is not False` rather than `is True`: evidence
        # recorded before the field existed carries the case count only at
        # the top level, and the fallback below reads it from there. What is
        # ruled out is a recording that KNOWS the two runs covered different
        # work — a ratio between a 132-case run and a 96-case one is
        # arithmetic, and the number alone does not show which it was.
        "traced_pipeline": tp.get("status") == "completed"
            and (tp.get("improvement_factor") or 0) >= 2.0
            and tp.get("runs_comparable") is not False
            and bool(tp.get("runs"))
            and all(r.get("serialised_frames_valid",
                          r.get("reflex_frames_valid"))
                    == r.get("cases", tp.get("cases"))
                    for r in tp.get("runs", [])),
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
    # Producer staleness over EVERY file this call read, in load order made
    # deterministic by sorting. JSONL evidence is exempt because a
    # line-oriented file carries no place to put a stamp; that is recorded in
    # the docstring of _read_lines rather than silently assumed.
    for _name in sorted(_LOADED):
        _stale, _unstamped = _producer_status(_name, _LOADED[_name])
        if _stale:
            stale_evidence.append(_stale)
        if _unstamped and _unstamped not in UNSTAMPED_GRANDFATHERED:
            unstamped_evidence.append(_unstamped)
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
    report = {
        "ok": True, "checks": checks, "missing_evidence": missing,
        "legacy_evidence": legacy,
        "unstamped_evidence": unstamped_evidence,
        "evidence_files_read": len(_LOADED),
        # The subject gap, in the only form that makes it a known quantity:
        # what IS bound, and everything that is not. `evidence_without_a_
        # subject` alone read 0 while 32 of 35 files had no subject, because
        # it silently required a `producer` stamp — and every unbound file
        # here is grandfathered, i.e. unstamped. A field meant to keep a gap
        # visible must not be able to report zero while the gap is the
        # normal case; the documented ratio is read off these two lists.
        # A file that exists nowhere is registered as {} and reported under
        # missing_evidence; it is not "unbound", it is absent, and counting
        # it here would inflate the gap with files nobody can bind.
        "evidence_with_a_subject": sorted(
            name for name, data in _LOADED.items()
            if isinstance(data, dict) and data and data.get("subject")),
        "evidence_without_a_subject": sorted(
            name for name, data in _LOADED.items()
            if isinstance(data, dict) and data and not data.get("subject")),
        # The narrower list the ratchet is about: stamped by a producer, but
        # never bound to what it measures.
        "stamped_without_a_subject": sorted(
            name for name, data in _LOADED.items()
            if isinstance(data, dict) and data.get("producer")
            and not data.get("subject")),
        "unstamped_grandfathered": sorted(
            name for name in _LOADED
            if name in UNSTAMPED_GRANDFATHERED
            and _LOADED[name] and not _LOADED[name].get("producer")),
        "layering": {"modules": layering["modules"],
                     "runtime_edges": layering["runtime_edges"]},
        "tests_evidence": ("executed now" if run_tests else
                           "tests-20260920.json" if test_run else
                           "architecture-20260917.json (stale: no digest, "
                           "predates the code it covers)"),
    }

    failed = [name for name, ok in checks.items() if not ok]
    # Unstamped evidence fails. It used to be reported only when something
    # else was already red, which made the report cosmetic: a file nobody can
    # tie to a producing script is not evidence, whatever it says.
    if failed or stale_evidence or unstamped_evidence:
        lines = []
        if failed:
            lines.append("architecture gate failed: " + ", ".join(failed))
        if stale_evidence:
            # Fail-closed: evidence whose producer moved on is not evidence.
            lines.append("stale evidence: " + "; ".join(stale_evidence))
        if layering["violations"]:
            lines.append("architecture violations: "
                          + json.dumps(layering["violations"]))
        if unstamped_evidence:
            lines.append(
                "evidence without a producer stamp (cannot be checked against "
                "the code that made it, re-run through research/evidence.write; "
                "pre-2026-09-20 files are listed in UNSTAMPED_GRANDFATHERED "
                "and are exempt until re-recorded): "
                + ", ".join(unstamped_evidence))
        if legacy:
            lines.append("legacy evidence accepted: " + "; ".join(legacy))
        if missing:
            lines.append(
                "missing evidence (" + str(len(missing)) + " file(s) found in "
                "neither " + " nor ".join(str(p) for p in dirs) + "): "
                + ", ".join(missing))
            lines.append(
                "a missing file is not a regression - re-run the benchmark, "
                "or check its output in under research/runs/ so a clone can "
                "reproduce this verdict.")
        report["ok"] = False
        report["failed"] = failed
        # The structured report travels WITH the refusal. A red gate had only
        # a text message, so anything that needed a field from it — a caller,
        # a test, a person — had to recompute the gate's own logic, and a
        # reimplementation is not a check of the thing it reimplements.
        stop = SystemExit("\n".join(lines))
        stop.report = report
        raise stop
    return report


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--run-tests", action="store_true",
                         help="run the test suite now instead of reading a "
                              "recorded result")
    _args = _parser.parse_args()
    print(json.dumps(verify(run_tests=_args.run_tests), indent=2))
