# Persistent ANN benchmark

Measured on `xrserver`, CPU path, 50,000 vectors, 64 dimensions, 100 queries,
same cosine data and top-k=10:

| Index | p50 | p95 | p99 | Recall@10 |
|---|---:|---:|---:|---:|
| NumPy exact scan | 5.464 ms | 6.592 ms | 7.017 ms | 1.00 |
| HNSW (`hnswlib`, ef_search=64) | 0.072 ms | 0.128 ms | 0.142 ms | 1.00 |

HNSW is about **76x faster at p50** and **49x faster at p99** on this workload.
This is a local index benchmark, not a claim about distributed recall or a
100B-document namespace. The exact path remains available as a recall oracle.
The HNSW index is persisted and reloads without rebuilding.

The production selection rule is now explicit: use HNSW for unfiltered dense
retrieval, retain exact fallback for filtered/recall-audit queries, then apply
metadata/ACL gates and the typed Pod ranker.

The integrated namespace materializer built a 10,000-row index in **656.05 ms**
on `xrserver`; this includes snapshot extraction, graph construction and disk
persistence.
