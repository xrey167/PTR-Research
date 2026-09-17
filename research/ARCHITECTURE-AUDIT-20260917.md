# Neural Pods architecture audit (17 Sep 2026)

## Implemented in this pass

- **Cache isolation:** result keys now include namespace, branch, Pod type, tags,
  canonical filters, vector digest, principal, generation and rank profile. This
  prevents ACL, generation and query-vector cross-talk.
- **Stampede control:** concurrent misses for the same key are coalesced into a
  single backend read. Cache statistics expose in-flight work.
- **Namespace revisions:** every schema/write/partial update/delete changes a
  monotonic namespace revision. Filter/posting/block indexes are invalidated for
  the whole namespace, including copy-on-write descendants.
- **Parallel multi-query:** independent query clauses fan out with a bounded
  executor and are merged deterministically with RRF.
- **Pod protocol:** `PodRequest`, `PodResponse` and `PodTransport` carry trace,
  deadline, hop budget, visited Pods, principal, generation and epoch. The same
  envelope can be adapted to Unix sockets, socat, gRPC, vLLM or a remote tunnel.
- **Transport hardening:** optional HMAC signing, target manifest-hash checks and
  bounded retries for transient connection/timeouts. Expired/cyclic requests
  fail closed.
- **PostgreSQL control plane:** the optional adapter now has namespace revisions,
  branch metadata, tombstones and revision-aware namespace inspection. It remains
  optional because this environment's measured path is SQLite.
- **Persistent dense retrieval tier:** `PersistentVectorIndex` provides a
  NumPy-vectorized, on-disk exact fallback with a stable interface for replacing
  it with FAISS/HNSW later. It is deliberately not mislabeled as ANN.
- **HNSW acceleration:** when `hnswlib` is available, `HNSWVectorIndex` persists
  the ANN graph and reloads it without rebuilding; exact search remains the
  recall oracle for filtered and audit queries.
- **Parallel Pod fan-out:** `PodFanout` dispatches independent branches with a
  bounded worker pool and preserves request order for deterministic merging.
- **Typed multiplex/duplex:** branch packets and event sequencing live in
  `pod_streams.py`; heterogeneous Pods exchange typed evidence rather than
  incompatible hidden states.

## Remaining architecture work

1. Put PostgreSQL behind a connection pool for multi-process production and add
   a durable object-store snapshot/WAL compactor for scale-to-zero.
2. Add the socket/gRPC network adapter around the authenticated protocol; keep
   retry policy idempotency-aware and never retry expired or revoked requests.
3. Add a shared distributed cache only after the local key contract is stable;
   use namespace revision/ETag as invalidation, with local LRU as the fast tier.
4. Add ANN index persistence and a real embedding worker pool. The local backend
   currently provides deterministic cosine search and optional local embeddings.
5. Add benchmark gates for p50/p95/p99 latency, hit rate, recall@k, ACL leakage,
   stale-generation rejection and branch update visibility.
6. Keep raw Multiplex hidden-state exchange restricted to compatible model-family
   Pods. Across heterogeneous Pods use the typed branch protocol.

## Design defaults

- Filter before scoring; include ACL, status, generation and Pod type in the hard
  filter, then apply learned ranking as an advisory score.
- Namespace is the primary isolation unit; tags are indexed metadata, not a
  substitute for ACLs.
- Strong consistency is the default for writes and generation changes. Eventual
  reads may be enabled only for explicitly latency-sensitive workloads.
- Every remote hop preserves one trace ID and decreases the hop budget. Cycles,
  expired deadlines, stale artifacts and revoked Pods fail closed.

## Validation

The contract/cache tests run on the remote CUDA environment because the local
Windows Python installation does not include pytest or torch. The benchmark
should report both cold (un-pinned) and hot (pre-warmed) namespace timings.
