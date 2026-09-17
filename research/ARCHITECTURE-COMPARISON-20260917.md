# Architecture measurement and research comparison

## Measured on `xrserver`

| Gate | Result | Interpretation |
|---|---:|---|
| 2,000-row local search recall@5 | 1.00 | Correct on the sealed synthetic fixture |
| 2,000-row local search MRR | 1.00 | Target document ranked first |
| Sequential search p50 / p95 | 7.72 / 40.02 ms | Exact local scan, warmed process |
| 800 parallel searches, 8 workers | 91.42 QPS | One-host SQLite reference, not distributed capacity |
| Search episode p50 / p95 | 43.91 / 85.08 ms | Includes agent/tool loop |
| Manifest validation p50 / p95 | 0.0496 / 0.0559 ms | Lineage gate is not the hot-path bottleneck |
| Cache benchmark p50 / p95 / p99 | 0.174 / 0.184 / 0.213 ms | 2,000 rows, 200 queries |
| Cache hit rate | 98.5% | Namespace-revision-aware local cache |
| ACL leakage | 0 hits | Outsider received no protected rows |
| Branch visibility | pass | New branch row was immediately visible |
| Stale-generation rejection | pass | Old generation filter returned no rows |
| HNSW 50k vectors p50 / p99 | 0.072 / 0.142 ms | Recall@10 1.00 on 100 queries |
| PostgreSQL control-plane writes | 0.702 ms/write | 100 writes, isolated schema |
| PostgreSQL quorum writes | 1.81 / 1.96 ms p50/p95 | 3 durable tables, quorum 2/3 |
| PostgreSQL degraded write | 1.13 ms | One replica unavailable, quorum preserved |
| mTLS persistent Pod transport | 0.102 / 0.151 ms p50/p95 | 100 requests, HMAC + manifest validation |
| TCP Pod transport | 1,821 QPS | 2,000 requests, 8 workers, 100% success |
| TCP multi-process transport | 1,606 QPS | 2,000 requests, 8 client processes, 100% success |
| Windows → xrserver transport | 1,796.94 QPS | 1,000 requests, persistent session, 100% success |

Raw reports: `runs/architecture-audit-benchmark-20260917.json` and the output
of `research/architecture_benchmark.py`.

## Comparison with the research targets

SID-1 describes synchronous RL rollouts with 1k+ bursty searches per second,
iterative tool choice, hybrid ANN/BM25 search and document ranking. Our current
measurement validates the control-plane and local correctness properties, but
does **not** reproduce SID's distributed 1k+ QPS result over a 100B-document
corpus: the current search fixture is local and small. We now have the
PostgreSQL control plane, revisioned quorum adapter, persistent ANN index and
cross-host transport; the remaining scale test is a larger corpus with a
distributed ANN/FTS serving tier and sustained burst fan-out.

Qdrant's distributed deployment supports shards, replicas and Raft-based
topology; its multitenancy guidance recommends physical partitioning for high
cardinality tenants. That maps to our namespace-per-locality rule, with
`tenant`, `pod_type`, `domain`, `generation` and ACL as indexed hard filters.

Redis client-side tracking supports server-assisted invalidation, but connection
loss flushes client caches and invalidation races need explicit handling. Our
local cache therefore keeps namespace revisions in every key and uses single
flight; a Redis tier should be added only as an optional cross-process layer.

## What is demonstrated versus still open

**Demonstrated:** typed Pod contracts, lifecycle/generation checks, ACL gates,
copy-on-write branch visibility, deterministic hybrid retrieval, cache isolation,
stampede coalescing, trace/deadline/hop protocol and duplex event ordering.

**Open:** distributed ANN recall at 100B-scale, object-storage cold reads,
cross-host PostgreSQL placement/consensus, distributed cache invalidation,
and broad external multi-hop quality. Those are explicit next benchmark gates
rather than implied by the local result.

Sources: [SID-1](https://www.sid.ai/research/sid-1),
[Qdrant distributed deployment](https://qdrant.tech/documentation/scaling/distributed_deployment/),
[Qdrant multitenancy](https://qdrant.tech/documentation/manage-data/multitenancy/),
[Redis client-side caching](https://redis.io/docs/latest/develop/clients/client-side-caching/).

Batch NeoHorse throughput: GPU0 137.15, GPU1 156.71, aggregate 293.86 tok/s at batch 4.

Sustained NeoHorse batch run: GPU0 154.14 tok/s (p95 836.7 ms), GPU1 147.34 tok/s (p95 946.9 ms), aggregate 301.48 tok/s; max allocation 8,245 MiB per card.

Adaptive batcher microbenchmark: 9,558 requests/s, 10,000/10,000 correct, 317 batches, mean batch 31.55, zero rejects/errors/deadline drops.

Full NeoHorse Pod path (GPU0/GPU1, 256 requests each, batch limit 64): 778.29/856.39 tok/s, **1,634.68 tok/s aggregate**, 512/512 correct responses, zero errors/rejects/deadline drops.

Concurrent two-GPU full Pod path: GPU0 756.51 tok/s + GPU1 715.93 tok/s = **1,472.43 tok/s aggregate**, 512/512 correct, no errors/rejects/deadline drops.

TCP socket + NeoHorse model path: 282.03 tok/s with 128 requests and 32 concurrent clients, 128/128 correct, zero errors.

Persistent TCP Pod sessions: 452.13 tok/s versus 282.03 tok/s with per-request connections (**1.60x improvement**), 128/128 correct.

Windows ? xrserver ? GPU model: 250.92 tok/s over 64 requests / 16 persistent sessions, 64/64 correct, zero errors.

mTLS transport validation: Windows ? xrserver with 16 persistent sessions and the real NeoHorse model reached **253.96 tok/s** (64 requests, 64/64 correct, zero errors) with HMAC and manifest validation enabled. This is a measured secure transport path, not a proxy estimate.

## Colibri resource and request tracking

The runtime now records bounded request receipts (request/trace/pod/generation identity, phase timings, token counts and cache hits), probes RAM/VRAM/disk capacity with explicit reserves, and enforces activation leases with deterministic VRAM ? RAM ? disk fallback. Unknown capacity is rejected. The `ResourceBoundHandler` binds admission to the actual Pod call and releases leases on all exit paths. On `xrserver`, the wrapper sustained 73,596 no-op admissions/s; full model and transport figures remain in MODEL-PROBE.

PodTransport now records bounded request receipts when configured with `RequestTracker`. In-process fanout measurement: 10,000 requests, 25,121.9 requests/s, 10,000/10,000 correct, 0 errors, tracker p50 0.360 ms / p95 1.168 ms, no active receipts remaining.

## TiKV/PD/Raft-inspired placement and fencing

Added `neural_pods.placement`: a control-plane PlacementDriver tracks node heartbeats, load and failure domains, assigns Pod regions across distinct zones, routes writes to a healthy leader, routes reads to healthy replicas, and rebalances after failure. `FencedLeader` provides monotonic terms and fencing tokens so stale writers cannot commit. This is deliberately a placement/fencing layer, not a home-grown replacement for audited Raft consensus.

Measurement: 10,000 assignments plus leader routes reached **295,737.8 placement operations/s**, selected three zones for replication factor three, and maintained monotonic epochs. Full suite: **256 passed, 1 skipped**.
