# So — Neural Knowledge Fabric: R144–R146 synthesis

## Mission
Build a knowledge substrate that is used by a language model more like internal knowledge than prompt-injected RAG, while remaining addressable, editable, revocable, versioned, portable and auditable.

This round specifically attacks three blockers reported by the independent Pod prototype:

1. free learned linking generalized 0% to held-out value pairs;
2. lifecycle management cost ~73% overhead vs target <5%;
3. superiority over strong RAG and novelty vs Larimar / neural memory were unproven.

The results below are new tests performed in this continuation. They do **not** constitute pretrained-LLM validation.

---

## R144 — systematic Pod reasoning, not Pair→Answer learning

### Design change
The relation/linking network is no longer asked to learn arbitrary value-pair mappings. Facts remain in Pods. A single shared neural hop operator learns **how to follow an addressed relation**. The world itself is newly randomized every training step.

Training queries additionally excluded:

- four complete relation bigrams: `(0,3), (1,2), (2,1), (3,0)`;
- about 20% of traversed ordered value pairs using a deterministic hold-out partition.

Thus the test contains both relation compositions and value pairs that were never supervised.

### Five-seed result

| Test | Mean final accuracy | Minimum seed |
|---|---:|---:|
| seen | 98.9875% | 98.7813% |
| held relation bigrams | 98.9375% | 98.7813% |
| held value pairs | 99.0000% | 98.7813% |
| both held simultaneously | 98.8938% | 98.8438% |
| 8 hops (training <=3) | 99.1063% | 99.0000% |
| 16 hops (training <=3) | 99.0500% | 98.8750% |

### Interpretation
This closes the specific 0%-held-pair failure **for graph/pointer-style compositional knowledge reasoning**. It does not claim arbitrary unseen functions can be inferred. If an unseen pair relation is arbitrary and neither encoded in the Pods nor implied by a learned reusable operator, inference is information-theoretically underdetermined.

Design rule:

> facts live in Pods; stable reasoning operators live in the model; do not train a free pair-specific link MLP to stand in for general reasoning.

This is consistent with earlier C25 role/filler separation and R137 changing-world training.

---

## R145 — lifecycle fast path

### Problem
Repeated Python-level generation/dependency validation on every Pod use dominates lightweight inference. Lifecycle correctness should not require graph traversal on the read hot path.

### New runtime semantics

1. pin one immutable Snapshot Lease per request;
2. use one monotonic branch-epoch comparison per neural step for strict-current semantics;
3. address Neural Images by revision/snapshot and ABI;
4. key derived caches by snapshot/dependency digest, making stale rows unreachable after mutation instead of scanning dependencies on every read;
5. compile only the changed Pod row after edit/revoke.

### Timings
512 queries x 6 hops, CPU, one thread, interleaved benchmark:

- baseline median: 1.6669 ms;
- snapshot fast path median: 1.6665 ms;
- measured median overhead: -0.027% (noise-level).

Across five independently generated interleaved timing trials, fast-path overhead ranged from **-0.761% to +0.393%**, all below the 5% gate.

The deliberately old-style per-record check measured ~33.8% overhead in the same benchmark.

Mutation path:

- one-row recompile: 0.0284 ms/edit;
- full memory recompile: 0.1155 ms/edit;
- ~4.07x mutation speed-up.

A lease created before a mutation was rejected after epoch advancement; the new lease reproduced baseline inference exactly.

### Scope
This closes the <5% **hot inference lifecycle** target in this toy runtime. It does not imply <5% end-to-end overhead on a real pretrained LLM, distributed store, cryptographic audit log or networked multi-node deployment.

---

## R146 — strong-RAG boundary and memory hierarchy

A deliberately strong baseline was used:

- exact oracle address;
- no retrieval error;
- no text parsing;
- retrieved fact already returned as a structured object identity.

The only remaining cost is translating the fact into the model-specific neural representation on every read.

The Pod path uses the exact same decoder but reads an already compiled Neural Image.

Both achieved 100% accuracy.

| Batch | Resident Pod median | Oracle structured RAG median | Speed-up |
|---:|---:|---:|---:|
| 1 | 0.0110 ms | 0.0329 ms | 2.98x |
| 32 | 0.0145 ms | 0.0386 ms | 2.66x |
| 512 | 0.0500 ms | 0.0897 ms | 1.80x |
| 4096 | 0.3646 ms | 0.5194 ms | 1.42x |

A Zipf workload with an exact-RAG cold fallback gives the following resident-cache envelope:

| Resident Pods / 96 | hit rate | expected speed-up vs oracle structured RAG |
|---:|---:|---:|
| 4 | 49.7% | 1.49x |
| 8 | 62.1% | 1.70x |
| 16 | 73.9% | 1.97x |
| 32 | 84.9% | 2.30x |
| 64 | 94.8% | 2.70x |

### Architectural conclusion
Do **not** frame the system as replacing RAG universally.

Use a knowledge-memory hierarchy:

```
source documents / databases / RAG
            ↓
    canonical Semantic Pods
            ↓
   immutable Pod revisions
            ↓
 model-specific compilation
            ↓
     Neural Images (hot)
            ↓
 address-first resident reasoning
```

Cold miss -> strong retrieval/source read.
Hot fact -> resident Neural Image.

This gives a Pareto envelope rather than forcing a false RAG-vs-memory dichotomy.

---

## Revised architecture: Revision-Native Neural Knowledge Objects

Each mutable fact/object is split into:

1. **Stable Symlink Identity** — semantic address.
2. **Immutable Semantic Revision** — model-neutral source of truth.
3. **Neural Image(s)** — compiled, model/ABI-specific cache lines.
4. **Snapshot Lease** — which version universe an inference request observes.
5. **Reasoning Operators** — model knowledge of *how* to combine Pods, trained over changing worlds.
6. **Derived State policy** — persistent only if lifecycle-correctable by LOA/closure, otherwise generation-bound cache or replay barrier.

The intended analogy is closer to a versioned virtual-memory subsystem than to prompt RAG.

---

## What is actually new vs what is prior art

Do not claim novelty for any single item below:

- editable external neural memory — Larimar;
- selective forgetting — Larimar and later work;
- pretraining with external factual lookup — LMLM;
- external knowledge tokens with edit/forget/restore/composition — Knowledge Externalization;
- scalable neural KV editing — NeuralDB;
- neuralized RAG / document modules inside inference — NeuRAG;
- cross-model memory modality / transfer — MindBridge and XMemTransfer;
- memory snapshots and rollback — ChronoMem;
- group equivariance, Koopman lifts, provenance or MVCC individually — established ideas.

The remaining research candidate is the **joint execution model**:

> changing-world pod-native training + stable semantic identity + immutable versioned knowledge objects + model-specific neural compilation + snapshot-pinned Neural Images + address-first paging + lifecycle-correctable persistent neural descendants / closure firewall + cold RAG fallback.

This is a candidate systems/architecture contribution. A broad paper/patent prior-art audit and a real pretrained-LLM benchmark are still required before claiming novelty.

---

## Updated DoD gates

### Passed in synthetic/runtime prototypes
- changing-world facts cannot be solved by fixed fact memorization;
- systematic held-out value/relation composition for pointer-style reasoning;
- long-chain reuse of the shared reasoning operator;
- edit/revoke/restore/version snapshots in prior R137-R143 work;
- stale-generation rejection;
- lifecycle hot-path overhead <5% in R145;
- strong exact-retrieval cold baseline can coexist with faster resident compiled Pods in R146.

### Still mandatory before a breakthrough claim
- real pretrained decoder LLM;
- natural-language query -> Symlink resolution on real datasets;
- general reasoning benchmarks beyond graph traversal;
- direct Larimar / NeuralDB / NeuRAG / LMLM / standard RAG experimental baselines on the same model/data;
- 10k / 100k / 1M Pod scale with ANN/address routing;
- multi-user concurrent snapshot semantics;
- real token, latency, GPU memory and quality measurements;
- leakage attacks after revoke, including alternate linguistic paths;
- source-to-Pod ingestion and conflict/provenance resolution;
- broad novelty + patent search.
