# PostgreSQL control-plane benchmark

Measured on `xrserver` against the local PostgreSQL service using an isolated
temporary schema (created and dropped in one transaction):

- 100 upserts with JSON metadata and vectors: **70.23 ms total**
- average write cost: **0.702 ms/write**
- namespace revision after writes: **101**
- copy-on-write branch creation: passed
- branch metadata inspection: passed
- branch tombstone delete: passed
- schema cleanup: passed

This measures the control plane only. Retrieval remains in the local indexed
reader; PostgreSQL is the authority for revisions, branches, ACL metadata and
lineage. The next production step is pooling connections and moving immutable
retrieval snapshots to object/NVMe storage.
