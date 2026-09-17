# TiKV / PD / Raft adaptation (2026-09-17)

## What the references contribute

Raft-rs separates consensus, log, state machine and transport. TiKV stores
regions in replicated Raft groups, while PD records node/store/region health
and chooses placement. The relevant idea for neural-pods is the separation of
semantic Pod routing from durable placement and the use of fencing for stale
writers.

References:

- https://github.com/tikv/raft-rs
- https://github.com/tikv/tikv
- https://github.com/tikv/pd
- https://github.com/pingcap/tidb

## Implemented

`neural_pods.placement` adds:

- node registration with failure-domain (`zone`) metadata
- heartbeat and load tracking
- deterministic region assignment with zone diversity
- leader-only write routing and healthy-replica read routing
- epoch increments on placement changes
- rebalancing after node failure
- `FencedLeader` leases with monotonic term and fencing tokens
- rejection of stale-generation / stale-leader writers

This is a placement and fencing layer. It is not presented as a complete
Raft implementation; the durable log, state machine and network transport are
still separate interfaces. A production multi-host deployment should use
raft-rs or another audited consensus implementation for the replicated control
plane rather than a Python reimplementation.

## Measurement on `xrserver`

10,000 region assignments plus leader-route checks:

- **295,737.8 placement operations/s**
- three distinct failure domains selected for a replication factor of three
- epoch reached 10,000 monotonically
- leader fence validation passed

Tests: **256 passed, 1 skipped, 15 warnings**.

`PlacementDriver.reconcile()` now performs a PD-style health pass and reassigns every region with a failed/stale replica, incrementing the placement epoch. The new reconciliation test passes; full remote suite is **257 passed, 1 skipped, 15 warnings**.
