# Imported So evidence review — 15 September 2026

Five local archives were supplied from `C:\Users\ReyDa\Downloads` and
copied into `research/imported_evidence_20260915`. Their archive hashes and
extraction paths are recorded in
[`ARCHIVE-MANIFEST.json`](imported_evidence_20260915/ARCHIVE-MANIFEST.json).

## Evidence inventory

### R287 — Temporal Capability Port Fabric

`r287-evidence.zip` reports a frozen-backbone, typed-value mechanism in which
identity-to-value bindings are mutable port state rather than learned
pair-specific weights. Across seeds 11, 23, 47, 89 and 131 it reports 1.0 for
held-out entity/value pairs, late-bound aliases, online updates, pointer
chains, pointer updates, revocation and stale-generation mutation invariance.
New aliases and entity bindings used zero optimizer steps.

This is strong mechanism evidence for our Symlink direction. It does not prove
natural-language entity linking or pretrained-LLM quality. The implementation
decision is to keep aliases and current value bindings in lifecycle-controlled
port/catalogue state; the Qwen LoRA learns reusable operations and routing,
not a separate gradient update for every new alias.

### R290 — Quantized Generation Port Capsules

`r290-evidence.zip` freezes `Qwen/Qwen2.5-0.5B` and evaluates a compactness
mechanism. The best passing variant is `residual-k3-v3-g32`: 1.0 exact
holdout accuracy and 1.0 top-match against the exact capsule, with 5,376 bytes
per fact versus 24,576 bytes for the BF16 capsule (21.875%). The shared-anchor
variant is trained only on the first 22 words; the 10 future words receive no
gradient. The report explicitly remains `NOT_DOD`.

This supports a future generation-bound capsule format, but the current local
reader uses normal LoRA/KV paths. We do not silently treat the quantized proxy
as a validated replacement for the real reader.

### R144–R146 — Neural Knowledge Fabric

The archive reports:

- R144 systematic pointer reasoning: approximately 98.89% mean accuracy when
  both relation bigrams and value pairs are held out; 99.11% at eight hops and
  99.05% at sixteen hops. Facts remain in Pods and a shared hop operator is
  trained over changing worlds.
- R145 snapshot fast path: median lifecycle overhead -0.027% in the toy
  runtime, five-trial range -0.761% to +0.393%; partial mutation is about
  4.07x faster than full recompilation.
- R146 resident compiled Pods versus an exact structured-RAG oracle: 100%
  accuracy in both paths and 1.42x–2.98x speed-up for resident Pods across
  batch sizes. This is an oracle structured baseline, not a natural-language
  RAG comparison.

These results directly reinforce our `ExecutionManifest`, revision-keyed
caches, partial-update path and the cold-RAG/hot-Pod hierarchy.

### R211–R214 — Neural Knowledge VM

The VM archive provides proxy evidence for a model-neutral revision compiled
to multiple ABIs. R211 is explicitly rejected because long floating products
drift. R211b passes with stable orthogonal operators and reports 564x tree
update speed-up. R212 reports global ABI calibration from five generator
correspondences with zero per-fact gradients. R213 is an informational noise
sweep. R214 reports one canonical revision mapped to four ABIs, with a proxy
2,289x speed-up versus naïve per-model recompilation.

The integration rule is adopted: exact IDs, numbers, time and generations stay
in the typed lifecycle lane; only stable reasoning operators may be composed
in a neural lane. The proxy timings are not claimed for Qwen or production
hardware.

### CT1–CT3 checked tensor supplement

The checked-tensor package validates 36 saved mathematical cases, 1,152
original and 1,152 edited output cells, plus six structural negative controls.
The included audit passes. CT2/CT3 are correctly rejected as per-query cost
savings for the small project shapes. This is deterministic arithmetic
replay, not a cryptographic proof, sandbox, model validation or full DoD pass.

## Verification performed here

- All 16 imported Python files compile under the project virtualenv.
- The included CT audit passes: 36 cases, 1,152 + 1,152 independently checked
  cells, six structural controls.
- R144–R146 and R211–R214 scripts were not rerun successfully because their
  original helper modules (for example `r137_pod_native_fabric`) are not in
  the supplied archives. Their signed/hashed JSON reports remain preserved as
  supplied evidence, not newly reproduced runs.

## Remaining boundary

These archives materially improve the research basis, but they do not include
the complete private So/CQP1/J-Space source and weights, nor do they close the
real pretrained-LLM natural-language, 100k-fact, 4/8/16-hop and strong-RAG
acceptance gates. The project remains an assembled, evidence-backed prototype.

The imported mechanism has now been connected to the local Qwen3B checkpoint on
`xrey@xrserver`. FP32 passes the transported-cache/logit gate; BF16 currently
fails the precision tolerance after five writes, so BF16 is recorded as an
open precision issue rather than being silently accepted.


## Local replay added (2026-09-16)

The R211b, R212, R213 and R214 proxy experiments now run from the checked-in evidence directory on Windows. Their former `/mnt/data` output paths were replaced with script-relative paths. The replay passes R211b and R214, confirms zero per-fact gradients in R212, and completes the R213 noise sweep. These remain proxy/mechanism evidence and are not substituted for pretrained CQP1/J-Space weights.


The CT1?CT3 checked-tensor package was independently replayed locally as well: 36 rows, 1,152 original and 1,152 edited cells, and six structural controls passed. The product-cost checker was run in an isolated copy; its recorded TC1 cost gate remains false, so this is preserved as mechanism evidence rather than a production security claim.
