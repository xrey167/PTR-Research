# Vespa review for the local Pod search

Reviewed 15 September 2026 against the Vespa repository and official
documentation. No Vespa service or package was installed in this Windows
workspace and no Vespa benchmark is claimed here.

## What Vespa contributes

Vespa models searchable documents with schema fields and rank profiles. It can
combine lexical matching such as BM25 with vector retrieval through the
`nearestNeighbor` operator and tensor fields. Approximate nearest-neighbor
search uses an HNSW index. Retrieval can be constrained by boolean metadata
filters before ranking.

Ranking is explicitly phased: a cheap first phase can combine lexical,
dense and metadata features, a bounded second phase can run a more expensive
model, and a global phase can re-rank the merged results. This maps closely to
our Dragonfly router plus a later reader/reranker, but Vespa's rank profile is
an index-side scorer rather than our learned Pod identity.

Vespa also supports streamed, real-time document writes and partial updates.
That is useful for generation changes and revocation markers, provided the
lifecycle barrier remains authoritative.

## Fit with Neural Pods

| Vespa capability | Local design decision |
|---|---|
| Schema fields and filters | Keep hard fields (`generation_key`, `status`, ACL, entities, validity) in every search projection. Filter before scoring. |
| HNSW nearest-neighbor search | Adopt as the next self-hosted replacement for the exact vector scan in `local_search.py`; keep the same SearchAction contract. |
| BM25 plus dense ranking | Use as a first-stage candidate union. Let Dragonfly/Symlink features and Pod-type gates participate in ranking. |
| Phased ranking | First phase: cheap lexical/vector/metadata score. Second phase: bounded Pod reranker or Qwen reader. Global phase: only after the execution manifest is validated. |
| Partial updates and deletes | Project new generations and revocations, but do not make the index the source of truth. Registry closure and `ExecutionManifest.validate()` remain mandatory. |
| Distributed content clusters | Candidate for a Linux/AMD deployment once a local ANN benchmark and burst-load test establish the expected contract. |

## Required projection contract

An eventual Vespa document or any replacement ANN record should contain at
least:

```json
{
  "id": "artifact:A91",
  "knowledge_key": "knowledge:supplier:muller:x12:lead_time",
  "generation_key": "generation:G8",
  "origin_keys": ["src:sap:po_history:92831"],
  "parent_artifact_keys": [],
  "status": "active",
  "acl": ["internal"],
  "semantic_type": "fact",
  "entity_ids": ["supplier:muller-gmbh", "component:x12"],
  "aliases": ["supplier delivery time", "Lieferzeit Müller"],
  "embedding": "...",
  "content": "..."
}
```

The index may rank this projection, but it may not authorize it. A query result
must be checked against the immutable generation and artifact set captured in
the execution manifest before it reaches a model or answer cache.

## Decision

Vespa is a strong future scale target for the retrieval projection, especially
when exact scanning stops meeting the burst-QPS requirement. We do not add the
Vespa dependency now: it would add a distributed Java/C++ serving system to a
working local prototype and would make the current measurements incomparable.
The next implementation step is a pluggable local HNSW backend with the
existing metadata, provenance and manifest gates. A Vespa adapter can then be
validated against the same corpus, recall/MRR, latency, revocation and stale
generation tests.

Sources: [Vespa repository](https://github.com/vespa-engine/vespa),
[nearest-neighbor search](https://docs.vespa.ai/en/querying/nearest-neighbor-search.html),
[HNSW ANN](https://docs.vespa.ai/en/querying/approximate-nn-hnsw.html),
[phased ranking](https://docs.vespa.ai/en/ranking/phased-ranking.html),
[writing and partial updates](https://docs.vespa.ai/en/basics/writing.html).
