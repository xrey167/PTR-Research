# Dragonfly evaluation split

The learned Dragonfly path has two different claims and they must not share a
score:

1. **Alias-family generalization:** aliases are present during `fit_pod`, but
   questions and paraphrases are held out. This measures learned address
   behavior.
2. **Unseen-alias resolution:** the alias never occurs in training. This needs
   an independently trained entity linker or a deterministic hard metadata
   binding; a per-Pod Dragonfly adapter alone has no information-theoretic
   basis to identify a new alias.

The current 10-entity hard benchmark is an unseen-alias retrieval stress test.
Its `provenance_router` score of 1.00 is therefore a hard-metadata control,
not a learned Dragonfly score. The existing learned alias suite covers the
first claim: 17 tests pass, including alias expansion, negative discrimination,
reload, revocation and retraining after generation changes.

The standalone learned-address pilot adds 12 held-out formulations across six
entities: Top-1 routing `1.00`, mean negative score `0.0149`. It uses a
synthetic encoder and is a mechanism sanity check, not evidence of enterprise
language generalization.

The first real-Qwen3B hidden-state transfer was intentionally run as a falsifying
check. After correcting candidate-feature binding, the frozen-hidden-state
PodAddress head reached only Top-1 `0.25` with mean negative score `0.8504`.
This fails the routing gate and shows that the synthetic encoder result does not
transfer. The next model experiment must train a projection/LoRA router (and
use pooled hidden states) before claiming learned Qwen routing.

That projection experiment is now implemented in
`research/qwen_dragonfly_projected_gpu.py` and passes on the local Qwen3B:
12 held-out formulations, Top-1 `1.00`, reload Top-1 `1.00`, mean negative probability `0.00506`,
loss decreased from `1.847` to `0.0072`. This is the first real-Qwen learned-routing result, but
it remains a small synthetic alias-family pilot until the provenance benchmark
is expanded.

The next paper table must report both columns separately rather than merging
them into one headline number.

The same projection trained on Vela-1.0-Encoder-307M reaches Top-1 0.833 and mean negative probability 0.0265 on the same 12 formulations. Vela is a compact router baseline; Qwen3B currently wins this pilot.

An ablation found a pooling mistake in that first Vela result: mean pooling
gave `0.833`, while the ModernBERT first-token representation gives `1.00`
Top-1 and negative probability `0.00579` on the identical split. The Vela
router now defaults to first-token pooling; both values remain recorded.
