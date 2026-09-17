# Turbogrep review for the local Pod search

Reviewed 15 September 2026 against `turbopuffer/turbogrep`.

## What Turbogrep actually does

The Rust CLI chunks source files with Tree-sitter, embeds chunks through Voyage,
stores them in a Turbopuffer namespace, then performs ANN search and reloads
local file content for ripgrep-style output. Its speculative mode races search
against an index sync and retries if the index changed. It also provides seeded
sampling, content-sensitive chunk IDs and bounded embedding concurrency.

## Relevance to Neural Pods

| Turbogrep idea | Decision here |
|---|---|
| Tree-sitter structural chunks | Adopt for code/document ingestion later; Pod facts stay semantic compiler units |
| Path + line + content hash IDs | Adopt the content-hash principle, but use OriginKey/GenerationKey instead of leaking paths |
| Speculative search while syncing | Adopt as a future local branch refresh mode, guarded by an execution snapshot |
| Embedding concurrency limit | Already represented by bounded parallel local search; add explicit embedding limits when an encoder service is introduced |
| Seeded sampling | Useful for deterministic corpus audits and held-out sampling |
| Voyage/Turbopuffer API | Explicitly not adopted; the local backend has no external API dependency |

## Findings that matter for our design

Turbogrep's own chunk model marks absolute paths as a production privacy TODO
and keeps content local while only indexing vectors remotely. For our internal
knowledge requirement, this reinforces the rule that search projections must
carry canonical lineage and ACL metadata, while raw source text remains behind
the local lifecycle barrier.

The sync/search race is useful only if the result carries a snapshot. Our
`ExecutionManifest` supplies that missing lifecycle contract: a refreshed
branch may produce candidates, but a stale generation cannot reach a commit.

## Bounded local follow-up

The existing local backend already passed 2,000- and 10,000-row measurements.
The 10,000-row exact-scan result fell to 14.17 QPS, so an ANN implementation is
still required for scale. Turbogrep does not provide that implementation locally
because its ANN call is delegated to Turbopuffer. The next self-hosted step is a
pluggable ANN index behind the same metadata and execution-manifest gates.

Sources: [turbogrep repository](https://github.com/turbopuffer/turbogrep),
[search implementation](https://github.com/turbopuffer/turbogrep/blob/master/src/search.rs),
[chunker](https://github.com/turbopuffer/turbogrep/blob/master/src/chunker.rs).
