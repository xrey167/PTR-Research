# Frozen Definition of Done — DO NOT SHRINK

The project is **not complete** until the same system demonstrates all of the following together.

1. A real pretrained decoder LLM backbone (minimum serious gate: open ~0.6B, target gate: ~3B+) is frozen.
2. At least 100,000 facts unknown to that pretrained checkpoint are installed after backbone training with **zero per-fact gradient updates**.
3. Normal inference uses **zero knowledge-text tokens** and no document/chunk RAG path.
4. Natural-language direct QA, paraphrase and implicit-reference QA work from installed knowledge.
5. Multi-hop reasoning over installed knowledge works at 4/8/16 hops and composes installed knowledge with pre-existing parametric capabilities.
6. New program/operator compositions that were not seen during Knowledge-ISA training generalize.
7. Edit, revoke, restore, rollback, branch and concurrent version snapshots work without backbone retraining.
8. Stale neural images, caches, JIT artifacts and old generations cannot resurrect revoked knowledge.
9. Lifecycle hot-path overhead is <5% versus the same knowledge-native inference without lifecycle checks.
10. On the same backbone/hardware/knowledge, end-to-end hot-workload inference is faster than a strong production-quality RAG baseline; target >3x where knowledge is reused, without lowering quality.
11. Frequently used reasoning can be revision-aware JIT compiled for >10x speedup versus the non-JIT knowledge-native path, with exact invalidation after edits.
12. Same canonical knowledge snapshot compiles to at least two model ABIs with the same semantics.
13. Runtime is crash-consistent, snapshot-isolated, tenant-safe, content-addressed and reproducible.
14. Direct comparison against closest neural-memory/editing systems (Larimar, Knowledge Externalization, NeuralDB, NeuRAG, Context-Distillation latent memories, and relevant state/KV injection systems) demonstrates a distinct technical mechanism and a measurable Pareto advantage.
15. Novelty claim survives a focused paper + patent prior-art audit. No claim may rely merely on: external memory, editable memory, neural memory, hot swap, versioning, knowledge objects, knowledge compilation, no-RAG memory, or KV injection individually.

No synthetic sub-result changes this DoD. Synthetic experiments are mechanism evidence only.
