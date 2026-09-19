# SID / Wilson / Inkling adoption

This iteration adopts three ideas from today's sources: iterative parallel search with reward traces (SID-1), sentence and dependency aware context plus graph chaining (Wilson search engine), and explicit heterogeneous model or modality profiles (Inkling-Small).

## Implemented

`LocalSearchAgent.adaptive_policy` now emits narrow and broad BM25 actions in parallel, records tool counts, and records whether retrieval stopped on target completion. The policy is a deterministic teacher for later GRPO or imitation training; it is not claimed to be a trained SID model. Existing Pod metadata and lifecycle gates remain authoritative.

The Wilson design motivates sentence-level contextual enrichment and relation chaining already present in `contextual_retrieval.py`, `spacy_enrichment.py`, and the Pod graph fields. Inkling-Small motivates a model-Pod profile that declares modality and routing requirements without downloading its 276B/12B-active checkpoint into the 3090 test host.

## Source comparison

- [SID-1 / turbopuffer](https://turbopuffer.com/blog/reinforcement-learning-sid-ai): iterative tool use, parallel calls, reward on recall/rank/latency.
- [Wilson search engine](https://blog.wilsonl.in/search-engine/): sentence chunking, semantic context, statement chaining, graph updates, disk-backed scale.
- [Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small): multimodal sparse MoE with explicit deployment profiles.

## Boundary

Our current measurement remains local and synthetic. It demonstrates the mechanics and traces, not SID-level recall or Inkling-level multimodal quality.

## NGU extension

The paper on Never Give Up motivates `neural_pods.ngu_sampling.never_give_up`: solved rollout items stop consuming samples, while failed items are retried up to a bounded budget. This is a control-plane primitive for future asynchronous GRPO/search training, with no claim of reproducing the paper's benchmark.
