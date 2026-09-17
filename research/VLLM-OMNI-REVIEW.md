# vLLM-Omni review for Neural Pods

Reviewed 15 September 2026 against the upstream `vllm-project/vllm-omni`
repository. vLLM-Omni is an inference and serving framework for heterogeneous
multi-stage models. It provides pipeline stages, KV-cache management,
disaggregated execution, streaming and an OpenAI-compatible server. It is not a
replacement for our Registry, Pod lineage, Dragonfly routing or lifecycle
barrier.

## What fits our architecture

| vLLM-Omni capability | Neural Pods use |
|---|---|
| Heterogeneous pipeline stages | route context, math guard and reader as separate execution stages |
| Pipelined/disaggregated execution | overlap retrieval, Pod loading and reader inference when snapshots permit it |
| KV-cache management | serve a validated ReaderCapsule efficiently after the lifecycle snapshot is pinned |
| Streaming/API server | expose the final answer only after the commit barrier; stream provisional tokens as uncommitted |
| ROCm support | candidate serving runtime for an AMD Instinct host, subject to a real ROCm smoke test |

## Required adapter boundary

The vLLM-Omni runtime must receive a resolved, immutable execution manifest:

```json
{
  "generation_key": "...",
  "artifact_keys": ["vector", "dragonfly", "reader-lora", "capsule"],
  "origin_keys": ["..."],
  "reader_identity": "...",
  "snapshot": "..."
}
```

`neural_pods.execution_manifest.ExecutionManifest` now implements this local
boundary and tests it against stale generations and source revocation. The
runtime may schedule stages and batch requests, but it must not select a
stale generation, bypass ACL/revocation checks, or publish an answer directly.
The Registry validates the manifest before loading and again before durable
answer materialization.

## What is not yet claimed

- No vLLM-Omni package was installed in this Windows workspace.
- No ROCm/Instinct serving run has been executed.
- The current trained reader was validated with Transformers/PEFT on
  `xrey@xrserver`; its Pod registration is independent of vLLM-Omni.
- vLLM-Omni does not provide our provenance graph, evidence independence,
  surprise gate, typed math guard or revocation propagation.

## Next bounded experiment

Create a Linux/ROCm serving smoke test with one resolved ReaderCapsule and two
requests: one valid current generation and one revoked/stale generation. Measure
time-to-first-token, end-to-end latency, peak VRAM and rejection correctness.
Compare the same manifest through the existing local ReaderCapsule path. This
is an integration benchmark, not evidence that vLLM-Omni improves answer
quality.

Source: [vLLM-Omni repository](https://github.com/vllm-project/vllm-omni),
[quickstart](https://github.com/vllm-project/vllm-omni/blob/main/docs/getting_started/quickstart.md).
