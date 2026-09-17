# vLLM-Ascend / Hugging Face component integration

The vLLM-Ascend repository is an accelerator backend and configuration layer,
not a replacement for Pod lifecycle checks. We use its design as a serving
capability selected by a Model-Pod, while lineage remains in the registry.

Model-Pods now accept optional component references:

- `router_ref`: a semantic/task router (for example Vela or an intent model)
- `ranker_ref`: a reranker/late-stage scoring model
- `stage_role`: the pipeline stage this Pod fulfills (reader, router, ranker,
  generator, verifier)
- `parallel_strategy`: declarative TP/DP/PP or replica settings
- `cache_policy`: Pod-scoped KV/cache limits and eviction policy

These references are emitted in the runtime activation plan as `pod_router`
and `ranker`. For MoE Pods, the actual expert router remains separate as
`router_model`; this avoids conflating expert dispatch with semantic routing.

From vLLM-Omni we adopt the useful separation between a declarative pipeline
and per-stage runtime overrides. The Pod stores the intended stage topology;
the serving adapter translates it to vLLM/vLLM-Omni flags on the target
hardware. Cache settings stay Pod-scoped so one branch can be updated without
changing another Pod's serving budget.

Suggested per-Pod combinations:

| Pod | Router | Ranker |
|---|---|---|
| context | Vela encoder or category classifier | mmbert reranker / local TensorRankProfile |
| retrieval | Vela embedding or intent classifier | mmbert reranker; BM25/RRF remains deterministic |
| reasoning | intent classifier | fact-check or feedback detector |
| model | task/intent router | toolcall verifier/sentinel |
| math | lightweight deterministic type router | no neural ranker unless validated |

These are component references, not automatically trusted weights. Each must
be versioned and bound to the Pod's lineage before activation. A Hugging Face
model can provide a soft routing or ranking signal; it cannot override hard
Pod type, ACL, generation or revocation filters.

Source review: vLLM-Ascend commit `92995fbbf30301b6f4b702fdb375a888608c1e20`;
HF model family listing includes Vela, mmbert reranker, intent, fact-check,
toolcall and safety classifiers. No remote model was silently downloaded or
claimed as trained in this change.
