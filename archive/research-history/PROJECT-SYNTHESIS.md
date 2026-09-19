# Neural Pods / So: local research synthesis

The cross-source audit of imported chats, research notes and all run families
is recorded in [FULL-PROJECT-AUDIT-20260915.md](FULL-PROJECT-AUDIT-20260915.md).
The five newly supplied archives are reviewed in
[IMPORTED-EVIDENCE-REVIEW-20260915.md](IMPORTED-EVIDENCE-REVIEW-20260915.md).
The two latest shared chats are reviewed in
[SHARED-LINK-REVIEW-20260915.md](SHARED-LINK-REVIEW-20260915.md).

This is the local handoff for the work recovered from the two shared chats and
the locally available project files. It records what is actually joined and
measured, what is only a mechanism test, and what remains unproven. It is the
authoritative map for continuing the work on another machine.

## The assembled path

```text
source record
  -> OriginKey / KnowledgeKey / Generation
  -> SemanticCompiler + local Registry DAG
  -> Qdrant-style filtered retrieval and Dragonfly/Symlink representation
  -> trained address planner
  -> model-produced link proof
  -> generation- and reader-bound KV capsule
  -> Qwen reader answer
  -> lifecycle barrier and committed answer receipt
```

Every committed answer in the integrated BF16 run carries the planner proof,
link proof, canonical generation, retrieval artifacts, capsule and source
lineage. Registry revocation is transitive and selective. `ReaderCapsules`
extends that contract so a future trained reader artifact is also a direct
capsule parent and its effective LoRA identity is checked at compile, load and
decode boundaries.

## Evidence that is complete

| Component | Evidence | Result |
|---|---|---|
| Provenance and lifecycle DAG | `neural_pods/registry.py`, provenance fanout tests | Origin/knowledge/generation/artifact closure, selective revocation and stale commit rejection |
| Semantic compiler and filtered retrieval | `neural_pods/semantics.py`, `semantic_routing.py`, `neural_pods/local_search.py`, `neural_pods/search_agent.py`, `neural_pods/pod_types.py` | Hard metadata is separated from soft semantic enrichment; typed context/math/model Pods, filters, branches, cold snapshots and iterative parallel search are tested |
| Dragonfly/Symlink alias identity | `runs/dragonfly-alias-002`, `runs/embedded-symlink-002`, `runs/identity-transfer-002` | Alias links and value-generation transfer mechanisms measured on small real models |
| Address planner | `runs/planner-training-003`, `research/audit_planner_training.py` | Real 0.5B LoRA: 49/50 held-out routing decisions, 50/50 reload reproduction, 9/9 known dialogue routing |
| Linked capsule barrier | `runs/dialogue-capsules-bf16-002`, `research/audit_dialogue_capsules.py` | Audit passed: 127 nodes, 7 full logit pairs, 9 cases, six revocation controls |
| Reader numerical control | `runs/reader-bf16-control-001`, `research/audit_reader_precision.py` | BF16 reader-only control: 6/7 positive answers by manual content review; 43.30 minutes under paging |
| Reader training path | `research/train_reader.py`, `research/evaluate_reader.py`, `research/reader_capsule.py`, `research/reader_answer_guard.py` | Qwen 2.5 3B LoRA training completed on `xrey@xrserver`; raw test is 79/92 and the typed guard audit is 92/92. New held-out two-hop runs: test 4/4 raw and 4/4 guarded; dev 3/4 raw and 4/4 guarded. |
| Current regression suite | local `pytest` run | 172/172 tests passed, two known dependency warnings |
| vLLM-Omni serving review | `research/VLLM-OMNI-REVIEW.md` | Architecture fit documented; ROCm smoke test still open |
| Execution manifest boundary | `neural_pods/execution_manifest.py`, `tests/test_execution_manifest.py` | Stale-generation and revocation rejection tested |
| Local stack measurement | `research/benchmark_local_stack.py`, `runs/local-stack-benchmark-001/report.json` | 1.0 recall/MRR; 101.7 local parallel QPS on 2,000 synthetic rows |
| Scale probe | `runs/local-stack-benchmark-10000/report.json` | 1.0 recall/MRR; 14.17 QPS at 10,000 rows; exact scan becomes bottleneck |
| Turbogrep review | `research/TURBOGREP-REVIEW.md` | Structural chunking and snapshot-aware sync adopted as design inputs; external API rejected |
| Vespa review | `research/VESPA-REVIEW.md` | HNSW, BM25/vector hybrid, filtered phased ranking and realtime projection updates fit a future ANN/scale backend; Vespa itself not installed |
| Local Vespa-compatible projection | `neural_pods/local_search.py`, `neural_pods/ranking.py`, `tests/test_vespa_projection.py` | Realtime partial updates with revision CAS, immediate snapshot invalidation, metadata-gated tensor/ONNX-compatible second-phase ranking; full suite 185/185 plus 3 targeted projection tests |
| Continuous concept mixer | `neural_pods/continuous_concepts.py`, `tests/test_continuous_concepts.py` | CoCoMix-inspired gated Pod-concept injection with projection, deterministic alignment, masking, gradient coverage and a real Qwen decoder hook; lifecycle validation remains in Dragonfly/Registry |
| Public concept-Pod demo | `research/demo_concept_pod.py`, `research/DEMO-GUIDE-20260916.md`, `runs/concept-pod-demo-001.json` | Reproducible tiny-Qwen activation/revocation demonstration: active concept changes logits by 0.3476, revoked concept restores baseline with zero delta |
| Pod gallery demo | `research/demo_pod_gallery.py`, `runs/pod-gallery-demo-001.json` | Six questions across context, math and model Pod types with wrong-type control and exact revocation restoration |
| Paper benchmark scaffold | `research/PAPER-READINESS-PLAN-20260916.md`, `research/generate_paper_benchmark.py`, `runs/paper-benchmark-pilot-001.jsonl` | Frozen 300/100/300 provenance-labelled pilot split and explicit baseline/ablation/CI plan; not yet external-world QA evidence |
| Pilot baseline quality gate | `research/evaluate_paper_baselines.py`, `runs/paper-baseline-pilot-001.json`, `research/PAPER-BENCHMARK-FAILURE-ANALYSIS-20260916.md` | TF-IDF Recall@3 saturates at 1.00 because of template/entity leakage; result rejected and benchmark hardening required |
| Leakage-controlled pilot | `research/generate_hard_benchmark.py`, `research/evaluate_hard_benchmark.py`, `runs/hard-baseline-pilot-001.json` | 30 entities with ten held out and query aliases hidden from corpus text; TF-IDF Recall@3 0.86, random 0.02, controlled hard-metadata provenance router 1.00; oracle-style synthetic pilot only |
| Learned Dragonfly pilot | `research/evaluate_learned_dragonfly.py`, `runs/learned-dragonfly-pilot-001.json`, `research/DRAGONFLY-EVALUATION-DESIGN-20260916.md` | Six synthetic entities, 12 held-out formulations: learned Pod-address Top-1 1.00 and mean negative score 0.0149; mechanism sanity check only |
| Qwen3B Dragonfly transfer gate | `research/qwen_dragonfly_gpu.py`, `runs/qwen3b-dragonfly-pilot-002.json` | Frozen Qwen3B hidden-state PodAddress transfer fails (Top-1 0.25, mean negative score 0.8504); projection/LoRA router is required |
| Qwen3B projected Dragonfly gate | `research/qwen_dragonfly_projected_gpu.py`, `runs/qwen3b-dragonfly-projected-001.json` | Trainable pooled-hidden-state projection/prototype router reaches Top-1 1.00 on 12 held-out formulations; reload Top-1 1.00; mean negative probability 0.00506; loss 1.847?0.0072; synthetic alias-family pilot |
| Temporal Port Plane | `neural_pods/symlink.py`, `tests/test_port_plane.py` | Late-bound aliases/value handles with zero alias optimizer steps, current-generation resolution, revision CAS and anti-resurrection checks; 10 targeted tests |
| Vela domain gate | `research/vela_domain_gate.py`, `runs/vela-domain-pilot-001.json` | Vela 307M classifies broad economics/business/law examples strongly, but misclassifies supplier lead-time as computer science; advisory gate only |
| Trained Pod taxonomy router | `neural_pods/pod_taxonomy.py`, `research/generate_taxonomy_dataset.py`, `research/train_pod_router.py`, `runs/taxonomy-routing-balanced-003.jsonl`, `runs/pod-router-training-003.json` | Balanced 500-row set (100/type; 60/20/20 train/validation/test), held-out templates/entities, fixed seed; Vela GPU run reaches 0.72 type and 0.90 domain accuracy on both held-out splits |
| Joint Dragonfly taxonomy router | `research/train_dragonfly_taxonomy_router.py`, `runs/dragonfly-taxonomy-training-001.json` | Shared learned Pod address plus type/domain/semantic-role/intent heads; held-out Dragonfly top-1 0.80, type/role/intent 0.84, domain 0.80; lifecycle authority remains deterministic |
| Vela encoder Dragonfly gate | `research/vela_dragonfly_router.py`, `runs/vela-dragonfly-router-001.json`, `runs/vela-pooling-ablation-001.json` | 768-dim ModernBert/Vela encoder: mean pooling 0.833, first-token pooling 1.00 on 12 held-out formulations; pooling choice was a real implementation bug |
| Transported lifecycle token | `neural_pods/lifecycle_transport.py`, `research/transported_lifecycle_experiment.py`, `tests/test_lifecycle_transport.py` | 100-case float64 orthogonal mechanism gate: static inverse median error 3.8879, transported max error 1.78e-15; Registry generation/revocation binding now enforced |
| Real Qwen KV lifecycle gate | `research/qwen_lifecycle.py`, `research/qwen_lifecycle_gate.py`, `tests/test_qwen_lifecycle.py` | Five feature writes on a real tiny Qwen decoder cache; transported deletion matches expected cache (5.96e-8) and logits (2.98e-8) in Float32 |
| Qwen2.5-3B GPU lifecycle gate | `runs/qwen-lifecycle-gate-3b-fp32-001.json`, `runs/qwen-lifecycle-gate-3b-bf16-001.json`, `runs/qwen-lifecycle-gate-3b-bf16-master64-001.json` | RTX 3090, 36-layer local Qwen3B: FP32 transported path passes (relative KV 3.97e-7, relative logits 1.96e-6); direct BF16 fails (KV 9.95e-3, logits 8.29e-2), while an FP64 lifecycle master with one BF16 materialization passes with zero comparison error |
| Imported So evidence R287/R290/R144-R146/R211-R214/CT1-CT3 | `research/IMPORTED-EVIDENCE-REVIEW-20260915.md`, `research/imported_evidence_20260915/` | Port-state alias binding, quantized capsule, changing-world pointer reasoning, snapshot fast path, multi-ABI revision compilation and checked tensor replay imported; proxy boundaries preserved |
| New shared-chat architecture review | `research/SHARED-LINK-REVIEW-20260915.md` | Port Plane decomposition and transported lifecycle-token hypothesis integrated as next gate; chat-only algebra values remain unverified |

## Explicit limits

The complete external-project archive `So_QC_CR1_Arbeitsstand.zip` version 33,
its raw weights and the original compact CQP1/J-Space implementation are still
not locally available. Five supplementary evidence archives are now preserved
under `research/imported_evidence_20260915`; they add proxy reports and small
mechanism implementations but are not silently treated as that complete source
tree. Recovered chat fragments remain under `research/recovered-fragments-002`.
The existing KV capsules are a full-prefix experimental state, not the
historical compact state.

The integrated reader run is technically valid but not a quality pass. Its
German three-week question still returns the 27-day fact without the required
`No`; the answer quality gate remains open. The two `UNKNOWN` cases are correct
abstentions. The reader does not yet prove broad implicit multi-hop use of Pod
knowledge as internal knowledge.

The typed 3B reader LoRA training bundle is frozen at 372/96/96 train/dev/test
rows and 46 planned optimizer updates. The real Windows CPU preflight reached
the first BF16 backward pass but completed zero optimizer steps after more than
20 minutes of paging. The evidence is in
`runs/reader-training-preflight-001/host-limit.json`. This is a host-capacity
result, not a failed model design and not a trained adapter.

## GPU evidence and next execution

The bounded preflight and two full LoRA runs completed on the RTX 3090 host.
`runs/reader-lora-gpu-001` is the original reader; `runs/reader-lora-gpu-004`
adds typed-Pod curriculum examples. Both preserve the frozen base and change
only LoRA parameters. The original reader is measured on the complete held-out
test (79/92 raw); the typed answer barrier audit is 92/92. The typed curriculum
has a 24-case week diagnostic (16/24 raw, 24/24 guarded).

The next required work is integration, not another unbounded training run:

1. Register the selected adapter as a `model`/`lora` artifact with
   `schema=research-reader:v1`, full reader identity, adapter hashes and IO
   schema.
2. Compile fresh `ReaderCapsules` for that identity and rerun integrated
   dialogue, selective revocation, stale-generation and multi-hop audits.
3. Add held-out contradiction and implicit Pod-type cases. Keep raw model and
   guarded results separate in every report.

The selected typed curriculum adapter is now registered and lineage-checked in
`runs/reader-model-registry-004.sqlite3`; it is not yet wired into the older
integrated dialogue registry. The full research goal remains open because the
original private So/CQP1/J-Space archive is missing and broad learned
multi-hop behavior has not been demonstrated.


## Unified local project gate (2026-09-16)

`research/run_project_gate.py` is the single reproducible acceptance entry point for the assembled stack. It runs the full regression suite, the fresh linked-model audit, revocation and generation controls, and local R211b–R214 replays. Run `python research/run_project_gate.py`; the current result is recorded in `runs/project-gate-001.json` with all 20 checks passing. The gate retains explicit limits: proxy experiments remain proxy evidence, and the unavailable private CQP1/J-Space originals are not implied.


The existing 2-hop-trained Qwen-3B adapter was also evaluated on a newly frozen three-hop extension (supplier transport + customs + warehouse). It achieved **4/4 exact answers** across held-out entities and English/German formulations in a fresh CPU inference process. This is transfer evidence, not a new training run; the detailed report is `runs/multihop-three-hop-transfer-001.json`.


A second Qwen-3B continuation was trained on `xrserver-dev` (CUDA:0, RTX 3090) from the validated two-hop adapter. The three-hop curriculum used 16 optimizer steps; a fresh GPU reload scored **4/4** on held-out English/German cases. Artifacts: `runs/reader-lora-threehop-001/report.json` and `eval-report.json`.

## Current assembled retrieval/embedding path (2026-09-16)

## Leakage-controlled generalization screen (2026-09-16)

`research/generate_generalization_benchmark.py` now creates a held-out
three-hop retrieval screen with separate transport, customs, and warehouse
records. Test roots, aliases, component handles, and question templates are
held out. On 20 test questions and 120 corpus records, character TF-IDF
retrieval reaches Recall@9 **0.417** and retrieves the full three-document
chain on **25%** of questions. A deterministic provenance-resolved route
reaches 1.0, which measures metadata resolution rather than learned language
quality. This closes the earlier template-leakage flaw while leaving broad
natural-language reader generalization open.

`LocalSearchBackend` now exposes a turbopuffer-compatible local contract:

```text
typed source text
  -> local embedder (SentenceTransformer/Vela-compatible callable)
  -> namespace schema + vector sidecar
  -> native-style Embed ANN/kNN query
  -> nested lifecycle-aware filters
  -> BM25/ANN multi-query + RRF
  -> projected attributes / aggregates
```

Namespace metadata includes schema, embedding model/dimension, timestamps,
branch state, index state, pinning and read-only state. `reembed_namespace()`
performs explicit model migration, increments row revisions and invalidates
pinned snapshots. The checked-in run `runs/native-embedding-reindex-001` uses
the real local 384-dimensional encoder: 6 Pods, 4 filtered native queries,
recall@1 **1.0**, plus a versioned re-embedding pass.

The final gate is `runs/project-gate-001.json`: **20/20 checks pass**, including
the full test suite (**217 tests**), linked-model/revocation/generation audits,
J-Space proxy controls, imported R211b-R214 evidence, the selected Qwen reader
regression and the native embedding reindex. The end-to-end local assembled-stack demo also proves typed routing, native retrieval, and revocation removal in one run (`runs/assembled-stack-demo-001.json`). The full-replay reader ablation
is retained at `runs/reader-lora-fullreplay-001` as a negative result and is
not registered for serving.

An additional boundary continuation was trained on `xrserver-dev` from the
validated mixed adapter using 64 new arithmetic examples (unseen suppliers,
English/German weeks and buffer cases, 128 CUDA optimizer steps at 5e-6).
The candidate improved the full two-hop holdout to **86/96** while retaining
**4/4** on the three-hop holdout. It is generation-bound and revocation-tested
in `runs/boundary-reader-registry-001.json`; the remaining ten errors are
explicitly retained for further curriculum work.

The same boundary candidate passes the typed semantic answer barrier on the complete 100-case holdout: raw Qwen output is 90/100, while guarded output is 100/100. Raw and guarded scores are kept separate; the guard recomputes only trusted typed arithmetic before committing an answer.

## NeoHorse-style routing and serving evidence

The assembled stack now includes a capability-aware routing harness that records
model selection, tool calls, outcomes and verified capability feedback. The real
NeoHorse-1-4B checkpoint was loaded on `xrserver-dev` and measured at **21.85
tokens/s** in a Transformers probe. The live Qwen3.8-27B reference server on
the same host was measured read-only at **18.45 wall-clock tokens/s p50**;
both GPUs were occupied, so no SGLang/vLLM claim is made.

The 10,000-row local workload measured **15.54 QPS** with 8 parallel workers,
53.8 ms p50 search latency, and 1.0 recall/MRR. These are SQLite exact-scan
reference numbers, not distributed ANN results. The authoritative combined
evaluation is `runs/combined-evaluation-001.json`; the current project gate is
20/20 with 217 tests passing.
