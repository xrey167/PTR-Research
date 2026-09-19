# Colibri resource and request tracking adaptation (2026-09-17)

## Findings from the Colibri source

The useful design is a sequence of explicit, machine-readable observations:

* Every generation is correlated by a request id and the engine emits `ACCEPT`,
  `DATA`, `DONE`, `ERROR` and phase profile records. `DONE` includes prompt and
  completion tokens, throughput, cache hit percentage and RSS.
* The server keeps bounded rolling profile history (120 turns), plus current
  hardware, residency map and tier counts. This avoids an unbounded telemetry
  memory leak while preserving recent diagnosis.
* Persistent expert usage is sparse and identity-bound. It is written atomically,
  supports decay and refuses incompatible dimensions/engine identities.
* Placement is budget-first: available RAM/VRAM is measured, safety reserves are
  removed, then pinned/LRU residency is sized from the actual per-layer expert
  width. Unknown capacity is never spent.
* Disk work is classified from a pre-request snapshot, so demand reads and
  speculative reads can be compared without counting the request's own recency
  bump. Speculative eviction has a heat guard.

## Adaptation implemented

`neural_pods.resource_runtime` now provides:

* `HardwareSnapshot` and `probe_hardware()` for CPU, RAM, NVIDIA VRAM and disk;
  missing probes remain `None`.
* `ResourceBudget.from_snapshot()` with explicit RAM, VRAM and disk reserves.
* `ResourceGovernor` with thread-safe leases and deterministic VRAM → RAM → disk
  fallback. A lease is required before residency is admitted and must be released.
* `RequestTracker` with bounded lifecycle receipts, request/trace/pod/generation
  identity, phase timings, token counts, cache-hit flag, resource snapshots and
  p50/p95/max latency aggregates. JSONL export is atomic at the API boundary.

This is transport-agnostic and can be attached to in-process, TCP or vLLM
handlers without changing the Pod wire contract. It deliberately does not use
speculative prefetch by default; a later policy can consume the recorded history
with a heat threshold and deadline guard.

## Measurements on `xrserver`

Hardware probe: 2 × RTX 3090 (24 GiB each), 31 GiB host RAM. Probe latency was
**38.89 ms** including `nvidia-smi`.

The 10,000-request admission/tracking run produced:

| Metric | Result |
|---|---:|
| Requests | 10,000 |
| Tracking + admission time | 0.0865 s |
| Tracking/admission throughput | **115,566 requests/s** |
| Leases admitted | 10,000 |
| Completed / errors | 10,000 / 0 |
| Cache hits | 5,000 |
| Tracker p50 / p95 | 0.0024 / 0.0050 ms |
| Active records after run | 0 |

The full resource-runtime test file passes **4/4**. The broader remote suite
remains green after synchronization (`245 passed, 1 skipped` before this
addition; rerun the full suite after merging concurrent changes).

## Safety boundary

The governor is an admission and accounting layer; it does not claim that a
model can be loaded merely because a file fits on disk. The next integration
step is to attach leases to model-Pod lifecycle and to expose request receipts
from `PodTransport`/`PodSocketSession`, then benchmark warm/cold placement and
eviction under real GPU inference.

Re-run after the resource-runtime integration: remote suite **250 passed, 1 skipped, 15 warnings**. A 10,000-request concurrent lifecycle/admission run completed 10,000/10,000 with 0 errors at **29,139.7 request lifecycles/s** including a simulated 50 �s work interval per request; tracker p50 0.668 ms and p95 1.459 ms. The isolated accounting path without simulated work measured 115,566/s in the paired benchmark. Hardware probe: 38.89 ms including nvidia-smi. The benchmark is accounting/admission throughput, not model-token throughput.

The governor is now bound to the request path through `ResourceBoundHandler`. A lease is acquired before the handler/batch queue and released on success, exception, or timeout. On `xrserver`, 10,000 no-op admissions through this wrapper achieved **73,596 requests/s**, with 10,000 admitted, 0 rejected, and no residual leases. This is control-plane overhead only; model throughput remains the separate NeoHorse measurements.

The NeoHorse socket benchmark now supports `--resource-bytes`. When set, each admitted model request obtains a measured VRAM/RAM/Disk lease before entering the adaptive batch queue and releases it on every exit path. This makes hardware admission testable on the real model/socket path without changing the Pod wire contract.

## Full model + lease + mTLS integration

On 2026-09-17, the real NeoHorse-1-4B model was served on `xrserver` with:

```text
ResourceBoundHandler(resource_bytes=256 MiB)
? BatchedPodHandler(batch=64)
? PodTransport(HMAC + manifest hash)
? PodSocketServer(mTLS)
```

A Windows client used 16 persistent mTLS sessions and sent 64 requests with 16 generated tokens: **13.757 requests/s**, **220.112 tokens/s**, 64/64 correct, 0 errors. GPU0 peaked at 9,216 MiB allocated during the run; both GPUs returned to ~1 MiB after shutdown. This is the first measurement covering model inference and hardware admission together.

Admission rejection proof: the same real mTLS model server was configured with an impossible 100 GiB per-request lease. Eight persistent-session requests returned eight transport errors in 0.2325 s; no request entered model generation. After shutdown both GPUs returned to 1 MiB used and 0% utilization. This verifies that resource admission is an enforcement boundary, not passive telemetry.

## Transport-level request receipts

`PodTransport` now accepts an optional `RequestTracker`; every dispatch creates and closes a receipt, including manifest, signature, capability, timeout and handler-error responses. A 10,000-request `PodFanout` run on `xrserver` with token/cache fields returned 10,000/10,000 correct at **25,121.9 requests/s**, 0 errors, tracker p50 0.360 ms and p95 1.168 ms. The tracker retained zero active requests and 5,000 cache hits.

The real socket server now constructs `PodTransport(tracker=RequestTracker(...))`, so remote socket dispatches use the same bounded lifecycle contract as in-process calls. The model path remains wire-compatible; only the server-side receipt layer is added. The code was compiled on Windows and `xrserver`, and the suite remains at 252 passed / 1 skipped.

## Generation-bound Pod residency

`ResourceGovernor.activate(pod_id, generation, amount)` now owns a Pod's residency lease. A second activation of the same generation is idempotent; a different generation cannot replace it without explicit deactivation, and stale-generation deactivation is rejected. A 10,000-operation concurrent activation/deactivation run on `xrserver` achieved **54,741.0 residency operations/s**, activated all 10,000 Pods, and ended with zero used bytes and zero resident Pods.

## Model residency lease in the real server

The NeoHorse server benchmark now accepts `--model-resource-bytes` and calls `ResourceGovernor.activate("neohorse", "neohorse-v1", ...)` before loading the checkpoint. The 10 GiB residency reservation was combined with 256 MiB per-request leases, mTLS, HMAC and adaptive batching. A Windows run completed 32/32 requests correctly at **11.533 requests/s / 92.263 tokens/s** (8 generated tokens). On natural server shutdown, the tracker reported **32 completed, 0 errors, 0 active**, p50 510.652 ms and p95 1,283.266 ms; both GPUs returned to 1 MiB used. This proves model loading and request execution share the same generation-bound resource lifecycle.
