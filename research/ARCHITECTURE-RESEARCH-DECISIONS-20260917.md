# Performance research decisions

## Storage and namespaces

Use the namespace as the first isolation and locality boundary. Keep `tenant`,
`pod_type`, `domain` and `generation` as indexed metadata for secondary
filtering. For high-cardinality tenants, use physical partitions/shards rather
than forcing every query through a payload filter. This follows Qdrant's
multitenancy guidance and avoids a large shared filter bitmap.

PostgreSQL is the durable control plane: metadata, lineage, ACL, revisions and
WAL-backed updates. Search data should be read from immutable namespace
snapshots and local ANN/FTS indexes. Read replicas can consume WAL/logical
replication; writers remain single authority for generation transitions.

## Cache hierarchy

1. per-process LRU (implemented) for sub-millisecond hot hits;
2. optional Redis cache-aside for cross-process results and invalidation;
3. local NVMe/object snapshots for cold namespaces.

Redis client-side tracking is useful for key-level invalidation, but cache keys
must still contain principal, namespace revision, generation and rank profile.
Disconnects must flush the local tier. Query results that include ACL-protected
data should use opt-in caching or remain process-local.

## Communication

The Pod protocol preserves one trace ID, request ID, deadline, epoch, principal,
manifest/generation identity and hop budget. A future network adapter should use
mTLS, authenticated manifest hashes and bounded retry with exponential backoff.
Retry only idempotent reads; never retry expired, revoked or generation-mismatched
requests. Keep duplex event ordering (`server_event_seq`, `epoch`) separate from
retrieval result ordering.

## Retrieval and ranking

Hard filters run before learned scoring: ACL, status, generation, tenant and
required Pod type/tags. BM25/FTS and ANN are queried in parallel, then fused by
RRF or a calibrated rank profile. Add a reranker only after measuring recall@k
and tail latency; top-20 evidence is the default evaluation setting. Block
postings and batched iteration are preferred to per-document recursive work.

## Measurement gates

Every change must report p50/p95/p99, cache hit rate, recall@k, ACL leakage,
stale-generation rejection and branch update visibility. The local gate is
`research/architecture_benchmark.py`; production gates should replay the same
queries against the PostgreSQL/ANN deployment.

## Sources

- [Redis client-side caching](https://redis.io/docs/latest/develop/clients/client-side-caching/)
- [Qdrant distributed deployment](https://qdrant.tech/documentation/scaling/distributed_deployment/)
- [Qdrant multitenancy](https://qdrant.tech/documentation/manage-data/multitenancy/)
- [PostgreSQL documentation](https://www.postgresql.org/docs/current/)
