# Current construct review

Reviewed 15 September 2026 after the typed-Pod GPU run.

## Fixed in this review

1. **Concurrent local reads.** SQLite branch materialization and pin access are
   now synchronized. Vector inputs reject NaN/infinite values and regexes are
   compiled once per search.
2. **Stale branch pins.** A write to a parent invalidates pinned snapshots for
   every descendant branch. A pinned empty snapshot is now still honored.
3. **Status/ACL gate.** Explicit non-`active` records and revoked records are
   excluded before ranking; string and list ACLs are both handled.
4. **Search reward.** Episodes now record a latency score and include it in the
   reward, so a future learned policy is measured on recall, rank and speed.
5. **Typed execution.** Context, math and model payload contracts are validated
   before typed artifacts are created. Math answers have a separate deterministic
   guard and raw model outputs remain separately reported.
6. **Provenance independence accounting.** `Registry.independent_origins()`
   deduplicates shared OriginKeys across repeated claims and derived artifacts.
   This prevents agent fan-out from inflating source counts while keeping the
   stronger statistical-independence question explicit.

## Research ideas turned into next tests

The recovered project notes point to four concrete extensions that fit the
current contracts:

- **Held-out contradiction gate:** pair active context Pods with a conflicting
  value and verify that the surprise gate defers instead of selecting either
  value silently.
- **Typed multi-hop:** traverse a context relation into a math Pod, evaluate
  the typed expression, and reject the complete receipt when any visited
  generation is stale or revoked.
- **Alias-cluster transfer:** change only a fact value and require the same
  Dragonfly identity representation to resolve the new generation; changing
  aliases or cluster metadata must require retraining.
- **Burst search benchmark:** run parallel `SearchEpisode`s against a branched
  corpus and record recall, MRR, p50/p95 latency and reward. This is the local
  analogue of the recovered RL-for-search idea and does not use an external API.

First local measurement: [local-stack-benchmark-001/report.json](../runs/local-stack-benchmark-001/report.json)
achieved 1.0 recall/MRR on 50 episodes over 2,000 synthetic records, 101.7
parallel searches per second with eight local workers, 9.48 ms sequential
search p50, 41.4 ms episode p50 and 0.052 ms manifest-validation p50. This is a
correctness and local baseline only; it is not an ANN, distributed or ROCm
benchmark.

The 10,000-row run ([report](../runs/local-stack-benchmark-10000/report.json))
also kept 1.0 recall/MRR, but exact-scan throughput fell to 14.17 QPS and search
p50 rose to 50.9 ms. This measured degradation is the concrete reason an ANN
index and distributed serving tier are required before making scale claims.

These are implementation-ready experiment definitions; they are not counted as
passed until new sealed cases and measurements are added.

## Remaining design limits

- The local vector path is an exact scan, not an HNSW/ANN implementation. The
  interface is pluggable, but large-corpus recall/latency is not yet measured.
- SQLite WAL is a local durable store, not a distributed object-storage tier.
  Branching, cold reload and parallel readers are verified on one host only.
- The search policy is deterministic. `LocalSearchAgent` exposes the episode
  and reward contract, but no RL-trained search policy has been claimed.
- The Qwen reader still makes arithmetic mistakes in raw generation. The guard
  is a safety barrier, not evidence that the base model learned general math.
- Original private So/CQP1/J-Space sources are still absent; recovered fragments
  remain explicitly marked as incomplete.

## Verification

The complete local suite passes (`172 passed`, two known dependency warnings).
Changed modules also pass direct `py_compile`; the historical recovered-fragment
directory contains intentionally escaped/non-source snippets and is excluded
from that targeted compile check.
