# Pod/model training and lifecycle plan

Date: 2026-09-17  
Dataset: `podmodel_v1`  
Status: prepared, not trained

## 1. What is now prepared

`runs/reader-training-inputs-pod-model-curriculum` contains 12 independently addressable capabilities with 50 examples in each of `train`, `dev`, and `test` (600 rows per split, 1,800 total). Every row carries:

- stable `id`, `task`, `pod_type`, `model_track`, and language;
- canonical JSON `target` rather than free-form labels;
- `provenance.origin_key`, dataset version, and source;
- assessment fields for capability, Pod type, model track, and example index.

The source data is synthetic and schema-focused. It teaches contracts and lifecycle behavior; it must not be presented as private company knowledge.

## 2. Dataset families and retraining boundaries

| Family | Pod type | Model track | Training objective | Retrain boundary |
|---|---|---|---|---|
| `router_function` | router | FunctionGemma-270M | strict tool JSON | router adapter only |
| `reader_sft` | reader | Qwen3-4B-SFT | grounded answer + citation | reader adapter only |
| `pod_linking` | transport | Qwen3-SFT | attested SSH/tunnel handshake | link adapter only |
| `multihop` | research | Qwen3-SFT | provenance-complete chains | research adapter only |
| `parallel_search` | research | Qwen3-GRPO | parallel branches, join, barriers | RL adapter only |
| `preference` | alignment | Zephyr-DPO | choose current/grounded/compliant output | preference adapter only |
| `reasoning_rl` | reasoning | Qwen3-GRPO | correctness/evidence/latency rewards | reasoning adapter only |
| `quantization` | model | LoftQ/QAT | preserve quality under quantization | quantization recipe/artifact |
| `vision_ocr` | vision | PaddleOCR-1B | structured OCR + privacy boundary | vision adapter only |
| `moe_routing` | router | MoE-experts | capability routing/load balancing | router/expert adapter |
| `lifecycle` | lifecycle | all | activation/revocation decisions | lifecycle gate adapter |
| `evaluation` | evaluation | all | frozen protocol and measurement | eval harness, never training |

To retrain one area, filter by `task` and `model_track`, create a new adapter/artifact, and keep the same `pod_id` while incrementing `generation`. Other Pods remain valid.

## 3. Data pipeline

1. **Ingest** raw documents, traces, tool calls, OCR pages, and Pod-link events.
2. **Normalize** into typed records: question, evidence, target, provenance, ACL, generation, and timestamps.
3. **Privacy gate**: local redaction/anonymization before any external model; retain the original `origin_key` in a protected registry.
4. **Semantic compiler**: resolve entities, aliases, canonical `knowledge_key`, relations, and retrieval tags.
5. **Split** by source/provenance, not random rows. Keep calibration, train, dev, test, and hidden adversarial holdouts disjoint.
6. **Generate task views** from the same canonical records: SFT conversations, preference pairs, GRPO rollouts/rewards, router calls, and OCR/vision records.
7. **Validate** JSON schema, required tags, lifecycle state, no revoked evidence, no test leakage, and deterministic hashes.
8. **Embed/index** only after validation. Store namespace, Pod type, tags, generation, ACL, and provenance as filterable metadata.
9. **Train** the smallest affected adapter/model track.
10. **Evaluate** on frozen and hidden sets; publish metrics and artifact manifest.
11. **Activate** only through the lifecycle barrier; branch the namespace before corpus changes.

## 4. Training order

### A. Continued pretraining (optional, domain adaptation)
Use clean, deduplicated domain text only. Do not mix task labels or evaluation questions. Produce a new base checkpoint with corpus hash, tokenizer, context length, and contamination report.

### B. SFT
Train the reader, router, link, multihop, and OCR adapters separately. Use completion-only masking where applicable, held-out provenance families, and structured JSON targets.

### C. Preference optimization
Use DPO/ORPO/KTO only for pairs with a clear preference signal: current generation, valid citation, ACL compliance, and useful answer. Keep rejected examples explicit so the model learns the failure mode.

### D. RL
Use GRPO/GSPO for measurable rollout behavior: multi-hop retrieval, parallel search, latency, tool efficiency, and anti-reward-hacking checks. Reward must include correctness and evidence, not just short outputs. Run memory-efficient RL only when the serving/training stack supports it.

### E. Quantization and serving
Use LoftQ initialization for quantized LoRA experiments and QAT where activation/weight quantization needs adaptation. Calibration data is separate from test. Export BF16/adapter as the source artifact; GGUF/dynamic GGUF is a serving derivative, never the training source.

## 5. Pod-to-Pod link contract

Every link training example and runtime request must include:

```json
{
  "trace_id": "...",
  "source_pod_id": "pod:research",
  "target_pod_id": "pod:reader",
  "target_generation": "g8",
  "artifact_id": "artifact:reader:g8",
  "transport": "ssh_tunnel",
  "capability": "grounded_answer",
  "acl": ["project:so"],
  "deadline_ms": 1500,
  "hop_budget": 3,
  "visited": ["pod:research"],
  "attestation": "manifest-hash"
}
```

The receiving Pod verifies identity, generation, artifact, ACL, revocation state, deadline, hop budget, and cycle-free `visited` list. A failed remote link falls back to a local Pod only when the policy allows it. WireGuard/mTLS can replace SSH later without changing the semantic contract.

## 6. Model/Pod lifecycle

`candidate -> trained -> evaluated -> approved -> active -> superseded -> retired/revoked`

`pod_id` is stable. `generation` changes on retraining. `artifact_id` identifies the exact adapter/checkpoint/export. A link points to the canonical Pod identity and resolves the active generation at invocation time. This preserves Symlink behavior: one semantic identity, many access paths, one lifecycle.

## 7. Evaluation and sizing

Every run records:

- retrieval: recall@k, MRR/nDCG, hybrid vs ANN/BM25;
- behavior: exact JSON/tool accuracy, grounded answer accuracy, multi-hop completion, provenance violations;
- systems: p50/p95/p99 latency, sustained/burst QPS, cache hit ratio, GPU memory, adapter size;
- lifecycle: stale/revoked rejection, ACL violations, cycle/deadline failures;
- quantization: task accuracy, KLD, memory and throughput delta.

Select the Pod tier from the measured Pareto frontier. Keep a frozen baseline and hidden holdout; never activate based only on training loss.

## 8. Source guidance incorporated

LoftQ jointly considers quantization and LoRA initialization, so it belongs in the quantization track: [LoftQ paper](https://arxiv.org/abs/2310.08659). The model-track split follows the Unsloth notebook catalogue and its FunctionGemma, Qwen/QAT, vision, SFT, DPO, and GRPO examples: [Unsloth notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks). Preference data follows the documented DPO/ORPO/KTO choices: [preference optimization](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto.md). Memory-efficient RL and FP8 are optional deployment/training modes and must be validated on the target GPU stack: [memory-efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl.md), [FP8 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning.md).

## 9. Immediate next run

1. Train `router_function` and `pod_linking` as separate adapters.
2. Train `reader_sft` and `multihop` separately.
3. Build preference pairs from held-out reader/multihop traces.
4. Run GRPO only on parallel/multihop cases with validated rewards.
5. Evaluate all adapters with the same frozen protocol, then activate one generation through the lifecycle barrier.


## 10. Additional Unsloth tracks to add next

- **Tool calling**: add a dedicated function-call view with valid/invalid JSON, unknown-tool rejection, argument-schema errors, and multi-tool sequencing. Keep tool names/versioned schemas in the target and evaluate exact tool/argument match.
- **Embedding fine-tuning**: build positive/negative query–Pod pairs from canonical aliases, entity/type/tag filters, and hard negatives from sibling Pods. Train the embedding/routing model separately from the reader; evaluate Recall@10, MRR and link-resolution accuracy. Never mix embedding dimensions or model families in one vector field.
- **NVFP4**: treat NVFP4 as a serving/quantization experiment with its own calibration and hidden holdout. Record kernel/runtime, activation scale policy, task accuracy, KLD, throughput and memory. Keep BF16 plus LoRA as the source of truth and only activate the NVFP4 artifact after parity checks.

These tracks extend `dataset_manifest.json` without changing the stable Pod identity. They are intentionally separate because tool-call correctness, embedding geometry, and low-bit numerical behavior have different labels and failure modes.

## Validierungsstatus

Die wiederholbare Prüfung liegt in `research/validate_pod_model_curriculum.py`. Sie kontrolliert Counts, Pflichtfelder, Provenienz, gültiges JSON, eindeutige IDs und disjunkte Splits. Die aktuelle Matrix mit dem Zweck jeder Pod-/Modellspur und der passenden Pipeline steht in `research/POD-MODEL-VALIDATION-AND-PIPELINE-MATRIX-20260917.md`.
