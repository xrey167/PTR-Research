# Continuous Concepts review (2026-09-16)

Source: [LLM Pretraining with Continuous Concepts](https://arxiv.org/abs/2502.08524).

The paper proposes CoCoMix: predict continuous concepts from a sparse
autoencoder and mix them with token hidden representations during training.
The useful fit for Neural Pods is direct: a Pod can carry a compact concept
vector while its hard provenance, generation and ACL metadata stay in the
symbolic lifecycle layer. This gives the model a learned semantic bridge
without turning the vector into authoritative metadata.

## Implemented mapping

`neural_pods.continuous_concepts.ContinuousConceptMixer` is a small model-side
primitive that projects variable-width Pod concepts, aligns them to token
positions, applies an explicit gate and supports masking. It is deliberately
additive rather than a full Qwen pretraining rewrite. The existing Dragonfly
router and Port Plane remain responsible for selecting and validating which
concepts may be activated.

## Integration plan

1. Semantic Compiler emits a concept target plus provenance/generation.
2. Dragonfly selects concepts only after lifecycle and ACL checks.
3. Qwen-LoRA trains the mixer/router and answer behavior jointly.
4. Revocation removes the concept from the activation set; no stale vector is
   allowed to bypass the Registry.

The paper's reported gains are not claimed for this repository. The local
tests cover shape, alignment, masking and gradient flow; a language-quality
claim requires a trained concept dictionary and held-out multi-hop evaluation.
