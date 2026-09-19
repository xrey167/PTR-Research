# Colibri integration review — 2026-09-17

Repository reviewed: `JustVugg/colibri` (source and runtime documentation at the
current repository head). The comparison is against the neural-pods stack in
this repository; no Colibri code is copied into the production path.

## Findings that directly apply

### 1. Tiered cache policy must be explicit and measurable

Colibri separates ordinary LRU, learned hot pinning, and routing-side cache
aware selection. Its defaults keep the model's router and precision unchanged;
lossy policies are opt-in and report route agreement, KL, swaps, and hit rate.
This is the right boundary for Pods as well. A cache hit must never silently
change the generation, artifact, ACL principal, or semantic route.

Implemented here: `PodCache.pin_query(...)` explicitly warms and pins one exact
cache key; `unpin(...)` restores the normal LRU bound. Pinning is excluded from
the default path, namespace revisions still form part of the key, and stats now
report `pinned` and `evictions`.

### 2. Persistent hot history needs identity and geometry checks

Colibri's `.coli_usage` records dimensions and engine identity and rejects
foreign histories. A Pod equivalent must bind hot history to namespace, branch,
pod type/tags, generation, artifact, encoder/model identity, and schema revision.
Our existing cache key and execution manifest already carry the critical
identity fields. Any future persistent hot-set format should use the same
authenticated manifest hash and reject a mismatched geometry rather than
silently applying another Pod's history.

### 3. Prefetch is useful only when it does not alter semantics

Colibri's PILOT prefetch predicts next-layer experts but keeps true routing
unchanged; it reports whether the prediction helped. For Pods, a safe analogue
is prewarming namespace snapshots, ANN pages, or model adapters based on
observed next requests. It must be bounded, cancellable, and measured with
prefetch hit rate, bytes read, p50/p95/p99, and cold/warm recall. Do not add
cache-aware route substitution to the default Dragonfly path.

### 4. Batch the boundary, not every internal operation

Colibri's mux protocol uses persistent slots, serial prefill, and continuously
batched decode. It emits request IDs, cancellation, per-turn telemetry, and
forward-compatible advisory lines. Our `AdaptiveBatcher`/`BatchedPodHandler`
already provides the corresponding compatibility key, deadline, and bounded
queue. The next serving improvement should add cancellation/queue age metrics
and preserve per-request IDs through the model adapter.

### 5. Segment boundaries require a typed state contract

Colibri's segment runtime makes state schema, numeric class, layer range, and
snapshot compatibility explicit, with transactional restore. This maps closely
to our typed Pod protocol: a remote Pod should advertise model family,
generation, artifact, manifest hash, state schema, numeric class, and capability
range before accepting a request. Raw hidden-state exchange should remain
restricted to compatible families; heterogeneous Pods should use typed branch
payloads.

### 6. Measurement discipline is part of the architecture

Colibri requires paired A/B runs, fixed prompts, cache-state disclosure,
correctness checks, and negative results. Adopt this for Pod optimizations:

* baseline and candidate in alternating order;
* cold and warm cache reported separately;
* p50/p95/p99 latency, throughput, recall@k, cache hit rate;
* ACL leakage, stale-generation rejection, branch visibility;
* output/hash equivalence for quality-preserving policies;
* quality-changing policies report route overlap/KL and are never enabled by
  default.

## What should not be copied blindly

Colibri's expert-level disk streaming, O_DIRECT, dual-SSD mirroring, and
model-specific kernels target very large sparse MoE checkpoints. They are not a
drop-in replacement for our Pod result cache or ANN index. We should reuse the
policy contracts and measurements, then implement storage-specific optimizations
only after a profile shows disk or model-weight movement is the bottleneck.

## Evidence from our current stack

The existing measurements provide the baseline for an A/B:

* local cache hit rate: about 98.5%;
* HNSW ANN p50: 0.072 ms with recall@10 1.0 on the current 50k-vector test;
* mTLS persistent Pod transport p50/p95: 0.102/0.151 ms;
* remote Windows-to-xrserver transport: 1,796.94 QPS;
* adaptive transport fan-out: 9,058 requests/s, 5,000/5,000 correct;
* two RTX 3090 sustained NeoHorse batch throughput: 301.48 tok/s aggregate.

The new pinning behavior is covered by
`tests/test_pod_cache.py::test_explicit_hot_pin_survives_lru_eviction_and_unpin_restores_bound`.

## Deeper runtime review

### Serving scheduler and cancellation

`docs/serve_protocol.md` documents a persistent mux with up to 16 KV slots,
serial prefill, continuously batched decode, request IDs, `STOP`, `CANCEL`, and
forward-compatible advisory telemetry lines. The important design property is
that cancellation is a normal terminal event and that a slot is persisted at a
safe boundary. A Pod must therefore distinguish queued cancellation from
inference cancellation and never report a cancelled result as committed.

Our `AdaptiveBatcher` now drops cancelled futures before model entry when
possible, cancels queued work when `BatchedPodHandler` hits a caller deadline,
and records queue wait and batch execution telemetry. A cancellation that races
with model entry is allowed to finish internally, but its Future is not fulfilled.

### Runtime/segment lifecycle

`docs/segment-runtime.md` requires explicit registration, model-family identity,
state schema, numeric compatibility class, half-open range, isolated sessions,
streamed snapshots and transactional rejection of corrupt or incompatible
restores. This reinforces our current rule that raw hidden-state exchange is
only allowed between compatible model-family Pods. For heterogeneous Pods, the
typed Pod branch protocol remains the correct boundary.

### Storage and expert placement

Colibri's `colibri.c` maintains a stable `(layer, expert)` residency index,
in-flight slot counters, bounded LRU victims, a separate pinned hot store,
one-layer-ahead prefetch, and weighted read-only mirror legs. The source also
validates mirror file sizes and safetensors headers and falls back to the
primary on mirror errors. These are weight-runtime mechanisms; our equivalent
safe adaptation is namespace snapshot prewarming and explicit result pinning.
We should only add weighted snapshot striping after measuring object-store/NVMe
read time as a bottleneck.

### Training implications

Colibri's tuning docs insist on fixed prompts, alternating paired runs,
teacher-forcing/output equivalence, cache-state disclosure, and rejection of a
candidate unless the throughput gain clears a threshold without quality or
tail-latency regression. This is directly applicable to Dragonfly and model-Pod
training: route substitutions or cache-aware policies need held-out alias,
multi-hop and ACL tests; lossless cache/prefetch changes need output/hash
equivalence; model-changing LoRA/quantization changes need recall and quality
gates.

## Fresh scheduler measurement after the adaptation

On `xrserver`, 10,000 requests through `PodRequest -> PodFanout ->
BatchedPodHandler` completed with:

| metric | value |
|---|---:|
| throughput | 9,142.89 requests/s |
| correct | 10,000 / 10,000 |
| batches | 318 |
| mean batch size | 31.45 |
| mean queue wait | 2.632 ms |
| maximum queue wait | 47.182 ms |
| mean batch execution | 0.114 ms |
| errors/rejects/deadline drops/cancellations | 0 / 0 / 0 / 0 |

The complete remote regression suite is **246 passed, 1 skipped**.

The batch boundary now also exposes cancellation and queue-wait telemetry:
cancelled queued futures are removed before model execution, and `stats()`
reports total and maximum queue wait in milliseconds. This preserves Colibri's
request lifecycle discipline without coupling the Pod protocol to a specific
model runtime. The current remote suite is **245 passed, 1 skipped**.
