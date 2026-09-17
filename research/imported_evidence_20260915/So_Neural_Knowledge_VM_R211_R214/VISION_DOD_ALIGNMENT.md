# Vision / DoD Alignment — R211–R214

## The vision is unchanged
Build a knowledge-native LLM architecture that is better than RAG for reused knowledge: knowledge behaves like internal model knowledge during reasoning, but remains separately installable, replaceable, deletable, versionable, branchable, auditable and portable across models.

The target architecture is **not** a vector database, not prompt RAG, not per-fact LoRA, and not a collection of cached answers.

## Frozen DoD — not reduced
The project is not done until the same system demonstrates all of the following together:

1. Real frozen pretrained decoder LLM, >=0.6B fast gate, target >=3B.
2. >=100k post-training facts, zero per-fact gradient updates.
3. Zero runtime knowledge-text tokens / no document-chunk RAG execution path.
4. Natural-language direct, paraphrase and implicit-reference QA.
5. 4/8/16-hop reasoning over installed knowledge and composition with parametric capabilities.
6. Unseen knowledge-program/operator compositions generalize.
7. Edit/revoke/restore/rollback/branch/concurrent snapshots without backbone retraining.
8. No stale neural/JIT/cache/generation resurrection.
9. <5% lifecycle hot-path overhead.
10. Same backbone/hardware/knowledge: faster than a strong RAG baseline, target >3x on hot/reused workloads without quality loss.
11. Revision-aware hot reasoning JIT >10x over the non-JIT knowledge-native path, with exact invalidation.
12. Same canonical knowledge snapshot compiles to >=2 real model ABIs with identical semantics.
13. Crash consistency, snapshot isolation, tenant safety, content addressing and reproducibility.
14. Direct comparison with closest neural-memory/editing systems.
15. Focused paper/patent novelty audit survives. No novelty claim may rest on external memory, neural memory, versioning, compilation, hot swap or KV injection alone.

No R211–R214 synthetic result satisfies or shrinks this DoD.

## What R211 changed
R211 attempted to place exact affine data and neural execution into one long floating operator product. After 4,096 operations, product-tree vs sequential evaluation differed by ~8.8e-10. This is rejected for authoritative truth.

**Design law:** Identity, number, time, generation, policy and other exact semantics stay in the typed exact lane. They are never made true by a long floating neural-product chain.

## R211b — stable neural algebra
Only norm-stable orthogonal neural operators are JIT-composed. Exact semantics remain outside that product.

Evidence:
- 100,000 post-training knowledge objects.
- two model ABIs (12D and 18D).
- up to 2,048 operator composition.
- cross-ABI semantic error remains ~1e-13 scale.
- 4,096-operation revision tree update is ~564x faster than full recomposition in this CPU proxy.
- product-tree vs sequential relative gap ~4.1e-15.

## R212 — global ABI calibration rather than fact training
A model ABI can, in an identifiable representation family, be solved from a small set of global generator correspondences using the intertwiner equations

    S_k P = P T_k

instead of fitting facts.

Evidence:
- only five global generator correspondences.
- one-dimensional intertwiner nullspace for both proxy models.
- solve time ~2–3 ms.
- unseen 256-step knowledge programs remain ~1e-14 scale.
- zero per-fact gradients.

This supports the desired architecture: calibrate a model once; install many facts afterward.

## R213 — calibration error does not compound per knowledge hop in the tested unitary representation
When the model ABI correspondence is noisy, deep program error is dominated by the global ABI misalignment instead of accumulating linearly with program length. In the tested orthogonal representation:
- 1e-3 generator noise -> ~6.4e-4 mean program error at length 256.
- 1e-2 generator noise -> ~4.9e-3 mean program error at length 256.

For exact orthogonal representations, this follows from the fact that an estimated global coordinate map conjugates the whole program; the error depends on coordinate misalignment, while the product remains norm-bounded.

This is promising for a real pretrained model only if a low-dimensional identifiable knowledge representation actually exists there. That remains unproved.

## R214 — revision once, execute on many models
The canonical knowledge revision is updated once in a semantic product tree; only the resulting root is mapped into each model ABI.

Proxy evidence:
- 8,192-operation program.
- 4 model ABIs.
- canonical update median ~0.034 ms.
- map updated root to all ABIs ~0.009 ms.
- rebuilding all four model programs separately ~100 ms.
- proxy speed ratio ~2,289x.
- old snapshot remains unchanged.

The important architectural point is not the proxy speed number. It is ownership: **the revision belongs to the model-neutral knowledge program; model-specific artifacts are rebuildable executions of that revision.**

## Integration with prior project results
R211–R214 do not replace earlier results. They refine them:

- C/E/R failures against free pair-linking, one-shot inverse injection, dense global dependencies and weak probe certificates remain binding negative evidence.
- C3–C14 language work remains the parser/planner basis; open-world knowledge IDs should be copied/resolved, not represented by an output class per fact.
- C53–C57 Qwen2.5-0.5B real-model runners remain the engineering bridge. The old design optimized a free residual per fact; the new path should replace this with a globally calibrated ABI / resident capsule mechanism.
- R187–R205 typed truth + stable operator basis + resident knowledge + structural lowering remains the executable layer.
- LOA/Closure/Generation results remain the persistent-descendant and anti-resurrection layer.

## Next non-negotiable gate
The next scientific gate is still real pretrained language-model execution, not another synthetic success:

**Qwen2.5-0.5B (pinned old C55/C56 revision) or another >=0.6B open decoder**
- backbone frozen;
- global ABI/socket calibration only;
- >=100k later-installed knowledge objects;
- zero fact gradients;
- zero knowledge-text runtime tokens;
- natural language / paraphrase / implicit / 4-8-16-hop;
- edit/revoke/restore/snapshot;
- fair strong-RAG comparison.

Until this passes, status remains: **mechanism-rich breakthrough candidate, not achieved DoD.**
