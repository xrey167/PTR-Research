# Full project and chat audit

Audited 15 September 2026 from the two locally imported shared-chat exports,
their message JSON files, all `research/*.md` handoffs, all run directories and
the current test suite. Chat claims are historical evidence unless a matching
local artifact and reproducible test exists.

## Chat material recovered

The imported chats under `research/imported-20260915/` cover the semantic
compiler, three semantic layers, Origin/Knowledge/Generation/Artifact keys,
provenance DAGs, selective revocation, Qdrant/Dragonfly/Symlink routing, typed
context/model/math Pods, Libelle conditional routing, LoRA readers and the
historical CQP1/J-Space/BCC1 direction.

The private project archive and original CQP1/J-Space weights are still absent.
Imported text therefore records hypotheses and historical claims, not proof of
those unavailable runs.

## Reproduced locally

| Area | Current evidence |
|---|---|
| Registry lifecycle | Immutable DAG, generation supersession, ACL/time gates, selective transitive revocation |
| Provenance | Shared-root fan-out and deduplicated `independent_origins()` |
| Semantic compiler | Hard metadata separated from soft enrichment; typed roles and relations |
| Local retrieval | SQLite-WAL BM25/regex/vector fusion, metadata gates, branches, pinning and parallel reads |
| Dragonfly/Symlink | Alias training, identity cluster checks, value-generation transfer and stale-link rejection |
| Typed Pods | Context/model/math contracts; deterministic math guard |
| Planner | Real 0.5B LoRA: 49/50 held-out decisions and reload reproduction |
| Reader LoRA | Real Qwen2.5-3B GPU training: 46 updates, frozen base, changed adapter; typed run registered as a model Pod |
| Reader evaluation | Complete test 79/92 raw; deterministic guard 92/92; typed-week diagnostic 16/24 raw and 24/24 guarded |
| Multi-Hop reader run | Qwen2.5-3B LoRA, 48 GPU updates; held-out two-hop test 4/4 raw and 4/4 guarded; dev 3/4 raw and 4/4 guarded; reports under `runs/reader-eval-multihop-*` |
| Integrated capsules | Lifecycle and lineage audits pass on staged runs; answer quality remains incomplete |

## Run status audit

Completed mechanism runs include `dragonfly-002`, `dragonfly-alias-002`,
`embedded-symlink-002`, `identity-transfer-002`, `linked-model-001`,
`planner-training-003`, `dialogue-capsules-bf16-002`, `semantic-002/003`, and
the GPU reader runs. `identity-transfer-001`, `semantic-001`, and earlier
planner runs are retained failures or stopped attempts, not current results.

Old `report.json` files still saying `running` are historical aborted runs
(`dragonfly-001`, `embedded-symlink-001`, CPU reader preflight); they are not
active processes. Their terminal outcomes are documented in the handoffs.

## Not proven by this audit

- broad implicit dialogue, follow-up resolution and multi-hop composition as
  learned model behavior;
- raw reader arithmetic/generalization without the deterministic guard;
- ANN-scale performance, 100B-document scale, 1k+ QPS or distributed storage;
- vLLM-Omni/ROCm serving on the AMD host;
- historical CQP1/J-Space/BCC1 results, the 100,000-fact target or superiority
  over strong RAG/frontier baselines;
- cryptographic signatures or remote attestation.

## Current acceptance gate

The project remains `assembled_not_complete`. Next decisive experiments are
sealed contradiction and implicit-context cases, typed context-to-math
multi-hop, integration of the registered reader into the dialogue registry,
and an AMD/ROCm serving smoke test with stale-generation rejection.

The machine-readable inventory is `research/project_inventory.json`.

The final audit record is `runs/full-audit-20260915.json`; the local performance
measurement is `runs/local-stack-benchmark-001/report.json`.
