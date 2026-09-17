# Completion audit against the assembled project goal

This matrix checks the requested integration against current files and saved
runtime evidence. It deliberately distinguishes a working mechanism from a
broader research claim.

| Requirement from the project work | Status | Evidence |
|---|---|---|
| Canonical identity, provenance and generations | **verified** | `neural_pods/registry.py`, linked-model audit, generation/revocation reports |
| Typed Pods for context, math, model and reasoning | **verified** | `neural_pods/pod_types.py`, taxonomy dataset/router, end-to-end demo |
| Learned Dragonfly aliases and Pod-type routing | **verified in held-out fixture** | `runs/dragonfly-taxonomy-training-003.json`, `runs/assembled-stack-demo-001.json` |
| Qwen LoRA carries value without evidence text | **verified in five-fact fixture** | `runs/multifact-internal-lora-004/report.json`, all test rows marked `fact_in_input=false` |
| Multi-hop training on a real Qwen3B | **verified in synthetic held-out fixture** | boundary/mixed adapter reports; three-hop 4/4 |
| Local Turbopuffer-like backend without provider API | **verified** | `neural_pods/local_search.py`, local backend tests and native reindex report |
| Native embedding, re-embedding and namespace metadata | **verified locally** | `runs/native-embedding-reindex-001/report.json` |
| Lifecycle revocation and stale-generation rejection | **verified** | linked replay, registry reports, execution-manifest tests |
| End-to-end routing → retrieval → revocation | **verified locally** | `runs/assembled-stack-demo-001.json` |
| Real-model Pod vs RAG timing | **verified on one RTX 3090** | `runs/multifact-internal-lora-004/pod-vs-rag-throughput.json` |
| Qwen3B GPU throughput | **verified on one RTX 3090** | `runs/reader-lora-boundary-002/gpu-throughput-report.json` |
| R211b–R214 mechanisms | **replayed locally** | imported evidence VM reports; proxy scope retained |
| 100B/distributed ANN throughput | **open** | current backend is exact SQLite scan; no distributed ANN run |
| General natural-language multi-hop quality | **partially measured, not established** | Leakage-controlled 20-query/120-document screen: TF-IDF Recall@9 0.417, local encoder 0.617, full-chain rates 0.25/0.30; base Qwen3B **0/20** vs mixed LoRA **17/20** raw and **20/20** after the typed Math-Pod barrier. Reports: `runs/generalization-benchmark-eval-001.json`, `runs/generalization-reader-eval-001.json`. Synthetic evidence only, not broad external-world QA. |
| vLLM/ROCm production-serving comparison | **open** | architecture reviewed, ROCm smoke test not completed |
| Full optimized RAG quality/latency/VRAM comparison | **open** | current timing isolates compact-vs-long prompt cost |

| NeoHorse-style capability routing harness | **verified locally** | `neural_pods/routing_harness.py`, `tests/test_routing_harness.py`; verified traces only feed capability feedback |
| NeoHorse-1-4B real checkpoint probe | **verified on xrserver-dev GPU** | `runs/neohorse-real-checkpoint-probe-001.json`; throughput probe, not official benchmark reproduction |
| Local 10k-row hybrid/metadata workload | **measured locally** | `runs/local-stack-benchmark-002/report.json`; 800 parallel searches, 15.54 QPS, p50 53.8 ms, recall/MRR 1.0 |
| xrserver live reference endpoint | **measured read-only** | `runs/xrserver-reference-serving-probe-001/report.json`; Qwen3.8-27B llama-server, p50 18.45 wall-clock tok/s; GPUs occupied |

## Current acceptance state

The separate generalization gate `runs/generalization-gate-001.json` passes
6/6 checks for held-out construction, semantic-vs-lexical retrieval, the real
Qwen baseline/adapter result, and the deterministic typed guard.

The reproducible project gate is `runs/project-gate-001.json`: **20/20 checks
pass**. The public-facing package is in `deliverables/`. The project is therefore
assembled and mechanically verified, while the open rows above prevent a claim
of general superiority or completed paper-grade validation.

## Historical failed experiments

A scan of the persisted reports finds seven older `status=failed` artifacts
from early semantic, Dragonfly, identity-transfer, embedded-symlink and
dialogue-capsule attempts. They remain as experiment history and are not
registered for serving. The acceptance gates use the later corrected artifacts
listed above, so an early failed run cannot be mistaken for the assembled model.
