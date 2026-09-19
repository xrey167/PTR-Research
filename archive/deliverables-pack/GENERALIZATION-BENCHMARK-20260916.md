# Leakage-controlled generalization benchmark

The earlier paper pilot was rejected because repeated templates and entity
surface forms made lexical retrieval artificially perfect. This replacement
holds out provenance roots, aliases, component handles, and question templates.

## Design

- 120 source records: 40 entities × transport, customs, and warehouse facts
- 20 test questions, each requiring all three records for a three-hop total
- adversarial distractors share procurement vocabulary
- deterministic provenance routing is reported separately from learned quality

## Result

| Method | Recall@9 | Full three-record chain |
|---|---:|---:|
| Character TF-IDF | 0.417 | 0.25 |
| Local 384-dim encoder | 0.617 | 0.30 |
| Provenance-resolved route | 1.000 | 1.00 |

The base Qwen3B reader scored **0/20** on the same evidence and questions. The
mixed multi-hop LoRA reader scored **17/20** in **4.28 s** on `xrserver-dev`
(RTX 3090). The three incorrect generations are retained in
`runs/generalization-reader-eval-001.json`; this is reader evidence on a
synthetic benchmark, not broad external-world QA.

The typed Math-Pod barrier recomputed the three unit-bearing values and yielded
**20/20 guarded results** for both readers. Raw generation and guarded commit
are reported separately: the barrier is a safety mechanism, not a claim that
the base model learned the task.

A focused 40-step continuation on these query forms was also tested. It scored
**16/20**, below the selected mixed adapter's 17/20, so it is retained as a
negative ablation and is not registered for serving. This guards against
choosing a later checkpoint merely because it was trained on the test-shaped
language.

The provenance row is an oracle-style metadata control, not a learned language
model result. The benchmark therefore demonstrates that held-out surface forms
break lexical retrieval while canonical lineage can recover the complete chain.
It does not establish broad natural-language Qwen quality; that remains the
next GPU experiment.

Reproduce with:

```powershell
.venv\Scripts\python.exe research/generate_generalization_benchmark.py
.venv\Scripts\python.exe -m research.evaluate_generalization_benchmark
```

Report: `runs/generalization-benchmark-eval-001.json`.

The previously trained boundary adapter was evaluated separately and also
scored **17/20** (4.22 s), with the same three failing IDs (`gen-23`, `gen-26`,
`gen-34`). The mixed adapter remains the selected artifact because it is the
already integrated lineage-checked reader; the boundary result is a tied
replication, not a reason to replace it.
