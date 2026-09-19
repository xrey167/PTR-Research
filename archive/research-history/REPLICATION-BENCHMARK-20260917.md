# Revisioned quorum benchmark

The new `QuorumReplicator` was measured with three independent replica objects,
quorum=2, 1,000 monotonic writes and a simulated failure of one replica:

- healthy write p50: **0.00564 ms**
- healthy write p95: **0.00588 ms**
- degraded (2/3 replicas) write: **0.00971 ms**
- latest revision/value after degradation: **1000**
- quorum remained available with **2 healthy replicas**
- majority-down case: fails closed with a quorum error

This verifies the revision/tombstone/quorum contract. The benchmark uses memory
replicas to isolate coordinator overhead; it is not a claim of distributed
consensus or durable replication. PostgreSQL/object-store replica adapters are
the next backend integration layer.
