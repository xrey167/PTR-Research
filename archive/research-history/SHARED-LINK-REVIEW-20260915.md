# Shared-chat review — 15 September 2026

Reviewed links:

- [Architektur Lösungen verfolgen](https://chatgpt.com/share/6aa9b9f6-5140-83eb-a32a-b5c1e619c6e4)
- [Quantenchat Fortsetzung](https://chatgpt.com/share/6aa9ba1e-c3d0-83eb-adc3-ee5bdea016a3)

## Link 1: Port Plane architecture

The conversation sharpens the current architecture into a separation between
stable language/reasoning capability and a live world-state plane:

```text
Language/Skill Transformer
        + Symlink Resolver
        + Temporal Pod Fabric
        + Compact Latent Port Codec
        + Position-Overlay Generations
        + Authority-Gated Port Attention
        + Independent Lifecycle Verifier
```

This is consistent with R287 and R290. New facts and aliases belong to
mutable, generation-bound port state; they are not per-fact weight edits. The
conversation explicitly keeps the result below a breakthrough claim until a
real transformer demonstrates full-logit equivalence, compact payloads,
exact lifecycle semantics, multi-hop behavior, low overhead and a fair RAG
comparison.

The chat also records a historical algebra experiment in which a static
inverse reportedly had a 25.03% median error across 100 random SO(8) cases,
while a later-write-conjugated token reportedly reached 8.77e-16 median error
and 1.24e-15 maximum error. These numbers are **chat-only** here: no runnable
artifact was supplied, so they are not counted as locally reproduced tests.

## Link 2: FIDT / Quantum-Linked continuation

The older continuation distinguishes two research lines and identifies the
later R64–R69 algebra/FIDT line as authoritative. Its key unresolved gate is
composition of multiple deletions in arbitrary order. A deletion token that is
simply appended as `m_k^-1` fails after later non-commuting writes; earlier
tokens must be transported through the later causal write frame. The proposed
next comparison is:

1. static adapter removal / inverse;
2. FIDT over recorded physical writes;
3. Symlink-transported lifecycle token.

The required measurements are fresh-rebuild equality, retain utility, forget
leakage, paraphrase and alternate-path access, and stale-state resurrection.
The linked chat does not provide a complete local source tree for this gate.

## Integration decision

The current repository already implements the safe part of this design:

- immutable Registry lineage and selective revocation;
- generation-bound Symlink/Dragonfly bindings;
- revision-checked partial updates;
- snapshot-bound `ExecutionManifest` validation;
- optional compact/tensor ranking behind hard lifecycle filters.

The explicit **Port Plane** object is now implemented in
`neural_pods/symlink.py`. It holds alias-to-current-generation bindings and
portable value handles, while the model learns only reusable routing/operation
behavior. Revision CAS, alias conflicts, stale snapshots and revocation are
covered by `tests/test_port_plane.py`. Transported delete/edit tokens are
implemented separately and still cannot bypass the Registry or manifest
barrier.

The isolated transported-token mechanism is also implemented in
`neural_pods/lifecycle_transport.py`. Its 100-case local report is
`runs/transported-lifecycle-local-001.json`: the static inverse has median
error 3.8879, while conjugated transport has maximum error 1.78e-15. This is
float64 orthogonal algebra evidence only; it is not a Qwen or production
neural-write result.

The first real-model gate now exists in `research/qwen_lifecycle.py` and
`research/qwen_lifecycle_gate.py`. It applies five feature-space writes to an
actual Qwen decoder `DynamicCache`, transports a middle delete, and compares
the result with the expected cache and next-token logits. The report is
`runs/qwen-lifecycle-gate-001.json`. It uses a tiny random Qwen configuration,
so it proves the cache/write contract rather than language quality. The same
gate was then run against the local Qwen2.5-3B checkpoint on `xrey@xrserver`:
FP32 passes with relative KV error `3.97e-7` and relative next-token-logit
error `1.96e-6`; BF16 exposes a real accumulated-precision boundary (relative
KV `9.95e-3`, logits `8.29e-2`) and is recorded as failed rather than hidden.
The mitigation is now measured on the same GPU: lifecycle writes run against
an FP64 master cache and are materialized to BF16 only once at the decoder
boundary. The five-write Qwen3B comparison then has zero KV and logit error;
the direct BF16 path remains rejected as unsafe for repeated edits.

No novelty claim is made from either chat. The Qwen result is a lifecycle/cache
contract gate, not evidence of language-quality improvement; multi-hop quality
and private research-weight integration remain separate experiments.
