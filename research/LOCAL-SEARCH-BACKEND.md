# Local search backend

`neural_pods.local_search.LocalSearchBackend` is the self-hosted search layer
for Pod retrieval. It is inspired by the operational goals in Turbopuffer's
description (bursty stateless reads, metadata filters, lexical/regex/vector
search, cold namespaces and branches), but it calls no Turbopuffer API and
does not require a hosted service.

## Data and lifecycle

- SQLite WAL stores namespace branches and Pod records durably.
- Branches are copy-on-write overlays; a child sees its parent until it writes
  or tombstones a key.
- `pin()` materializes a reader snapshot; `unpin()` releases it, so cold
  namespaces reload from disk on the next read.
- Readers are read-only and thread-safe. The backend can therefore serve
  parallel bursts from a pool without a per-request mutable session.
- `namespace_metadata()` exposes row count, timestamps, index state, branch
  parents and pinning/read-only state. `update_namespace_metadata()` provides
  the local write barrier and pinning configuration; it never bypasses Pod
  lineage checks.

## Ranking and safety

`search()` applies ACL, revoked/status and caller metadata filters before any
score is computed. It then fuses cosine similarity, BM25-style lexical score,
and regex matches. The implementation is an exact vector scan today; the API
leaves room for an optional HNSW/ANN index without changing lifecycle gates.

The backend is only a candidate index. `Registry.snapshot()` must still verify
OriginKey, KnowledgeKey, Generation and transitive revocation immediately
before a model sees a Pod or commits an answer. This preserves the rule:
**no neural state without lineage**.

## Query contract

`query()` mirrors the self-hosted subset of the turbopuffer Query API: a
`rank_by` expression can select ANN, BM25 or attribute ordering; filters accept
`Eq`, `In`, `ContainsAny` and nested `And`; and a root `queries` array executes
up to 16 subqueries against one snapshot and fuses them with RRF. Returned rows
carry `$dist` and only expose metadata that already passed ACL, revocation and
generation gates. This keeps the API useful for Dragonfly fan-out without
introducing a turbopuffer dependency.

Weighted `Sum` and `Product` clauses are also supported for local hybrid
ranking, allowing one query to boost exact BM25 evidence while retaining a
semantic ANN signal.

When `LocalSearchBackend` is constructed with a local `embedder` callable,
`configure_schema(namespace, {"text": {"embed": ...}})` enables native-style
embedding on writes and `query()` accepts `["Embed", text]` at query time.
This keeps embedding computation inside the deployment (for example the
local Vela/SentenceTransformer stack) and never sends source text to an
external provider.

`research/run_native_embedding_reindex.py` exercises this path with the
checked-in encoder: six typed canonical Pods are re-embedded on write and
four filtered native ANN queries achieve recall@1 = 1.0. The persisted result
is `runs/native-embedding-reindex-001/report.json`.

`neural_pods.search_agent.LocalSearchAgent` adds the iterative layer: a policy
can fan out BM25, ANN, hybrid and regex calls per turn, observe the merged
ranked candidates, reformulate, and stop early when all target artifacts are
found. `SearchEpisode` records recall, MRR, reward, calls and elapsed time, so
the same harness can later train a Dragonfly/Qwen policy with search rewards.
The current policy is deliberately deterministic and acts as the regression
baseline; it is not presented as an RL result.

The proxy-compression research reference supplied for this project is treated
as a future model-training input representation, not as a search dependency:
https://github.com/LZhengisme/proxy-compression
