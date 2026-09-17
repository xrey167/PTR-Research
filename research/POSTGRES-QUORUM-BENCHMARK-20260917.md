# PostgreSQL quorum benchmark — 2026-09-17

Run on `xrserver` using PostgreSQL 16 and three durable replica tables in the
`np_replicas` schema. The benchmark uses a quorum of 2/3 and writes one
revisioned Pod record repeatedly. The third connection is then closed and a
degraded write is measured with the remaining two replicas.

Command:

```bash
PYTHONPATH=. /srv/ai/workspaces/llm-lora/.venv/bin/python \
  research/benchmark_postgres_quorum.py
```

Result (100 writes):

| Metric | Result |
|---|---:|
| Replicas / quorum | 3 / 2 |
| Healthy write p50 | 1.80 ms |
| Healthy write p95 | 1.96 ms |
| Degraded write with one replica unavailable | 1.20 ms |
| Latest revision/value read back | 100 |
| Successful writes | 100% |

This validates durable revisioned writes, tombstone-capable records, highest
revision reads, and continued availability after one replica is unavailable.
The current benchmark places all three replicas in one PostgreSQL instance;
it is therefore a durable quorum adapter rather than a substitute for a
multi-host consensus system such as Raft. The benchmark intentionally leaves
uniquely named tables for post-run inspection, avoiding DDL lock contention
while connections are being closed.
