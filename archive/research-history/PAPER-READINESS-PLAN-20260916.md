# Paper-readiness assessment (2026-09-16)

## Current status

The repository is a tested systems prototype. The current tiny-Qwen demos prove
that Pod activation, type routing scaffolds, and revocation can change and
restore hidden-state behavior. They do **not** establish language-quality
improvement, generalization, or superiority over retrieval baselines.

## Falsifiable research claim

> A generation-bound continuous-concept memory with hard provenance gates
> improves multi-hop enterprise QA and makes revocation exact, compared with
> static RAG, cosine-only Pod retrieval, and an ungated concept adapter.

The paper must test both halves. A lifecycle result alone is an algebra/system
paper; a QA result without revocation is ordinary RAG/adapter work.

## Required benchmark

Build a versioned, provenance-labelled benchmark from the imported evidence and
synthetic enterprise records:

- 300 train / 100 validation / 300 held-out questions;
- 3-hop and 5-hop questions, with paraphrased aliases and distractors;
- context, math, rule/model and mixed Pod types;
- source-level independence labels so duplicated evidence is not counted twice;
- revocation/edit cases where one parent changes and unaffected answers must stay;
- private originals stay local; publish hashes, schemas and generators only.

## Required baselines and ablations

1. static BM25/vector RAG;
2. cosine-only Pod retrieval (the existing 16/18 baseline);
3. typed metadata + ANN without learned concepts;
4. ungated continuous concept mixer;
5. full system: Dragonfly routing + concept mixer + lifecycle barrier;
6. full system with one parent revoked, measuring selective closure.

## Metrics

- exact answer and calibrated answer accuracy;
- multi-hop chain recall and distractor rejection;
- alias/generalization accuracy on unseen spellings;
- independent-root evidence precision;
- revocation precision/recall and stale-answer rate (target: zero stale);
- latency, memory and QPS under burst load;
- bootstrap 95% confidence intervals and paired randomization tests.

## What may be claimed now

- transported lifecycle tokens outperform a static inverse in the controlled
  orthogonal-write experiment;
- the Qwen KV contract and FP64-master BF16 mitigation pass their current gates;
- concept activation and exact revocation are reproducible on tiny Qwen.

No claim of improved QA, multi-hop generalization, or production-scale recall
should be made until the benchmark and baselines above are run.

The deterministic pilot generator is available at
`research/generate_paper_benchmark.py`; it creates 300/100/300 train,
validation and test rows with provenance labels. These are a benchmark
scaffold, not external-world evidence and must be replaced or supplemented by
the approved private/real corpus before submission.

The first baseline run exposed template/entity leakage (TF-IDF Recall@3 =
1.00). That result is recorded as a failed benchmark-quality gate in
`research/PAPER-BENCHMARK-FAILURE-ANALYSIS-20260916.md`.

A corrected 30-entity pilot now holds out ten entities and query aliases. Its
character-TF-IDF baseline is Recall@3 `0.86` on 50 queries, with random at
`0.02` (`runs/hard-baseline-pilot-001.json`). A controlled provenance-router
ablation reaches `1.00` by using hard alias/entity/component metadata before
retrieval. This demonstrates the intended routing mechanism, but it is an
oracle-style ablation, not learned Dragonfly evidence. The pilot is still
synthetic and too small for a publication claim.
