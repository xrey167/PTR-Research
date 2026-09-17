# Adaptive Pod batcher — 2026-09-17

`neural_pods.adaptive_batcher.AdaptiveBatcher` provides model-runtime-neutral
micro-batching for Pod inference. Requests carry a compatibility key, so
different model revisions, adapters, principals, or pod types cannot be mixed
accidentally. The queue has bounded capacity and each request can carry a
monotonic deadline. Expired requests are completed with `TimeoutError` without
calling the model. Runtime errors are propagated to every request in the
affected batch.

## Measured control-plane overhead

Remote `xrserver`, 32 submitter threads, 10,000 requests, maximum batch 64,
2 ms queue window:

| Metric | Result |
|---|---:|
| Requests/s | 9,557.96 |
| Correct results | 10,000 / 10,000 |
| Batches | 317 |
| Mean batch | 31.55 |
| Maximum observed batch | 32 |
| Rejected | 0 |
| Deadline drops | 0 |
| Runtime errors | 0 |

The benchmark measures scheduler overhead and correctness. Model throughput is
measured separately in `MODEL-PROBE-20260917.md`; the practical GPU operating
range for NeoHorse is batch 64–128.

`BatchedPodHandler` now adapts this scheduler to the existing `PodTransport`
handler contract. A Pod can register it without changing callers; the
transport fan-out test passes with four same-principal requests in one batch
and a separate principal in an isolated batch. The full remote suite is
**241 passed, 1 skipped**.

End-to-end PodTransport benchmark: 5,000 requests through PodRequest, PodFanout and BatchedPodHandler completed in 0.552 s (**9,058 requests/s**), 5,000/5,000 correct, 0 errors, 160 batches, zero rejects or deadline drops.

After adding cancellation and queue telemetry, a fresh remote run reached
**9,142.89 requests/s**, 10,000/10,000 correct, 318 batches, zero
errors/rejects/deadline-drops/cancellations. Mean queue wait was **2.632 ms**
(maximum 47.182 ms); mean scheduler batch execution was **0.114 ms** and the
mean batch size was **31.45**. Raw data is in
`runs/batched-transport-colibri-20260917.json`.
