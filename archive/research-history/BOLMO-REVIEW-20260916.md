# Bolmo review (2026-09-16)

Source: [arXiv:2512.15586](https://arxiv.org/abs/2512.15586).

Bolmo converts subword language models to byte-level models with a two-stage
conversion and limited additional training. The relevant idea for Neural Pods
is the boundary between raw evidence and semantic compilation: byte-level
handling can preserve exact aliases, identifiers, anonymized fields and code
fragments before the Pod compiler creates canonical entities and lifecycle
keys.

This does **not** solve the BF16 lifecycle error directly. The measured issue
is repeated low-precision KV writes. Our tested fix is independent: perform the
short noncommutative lifecycle chain in an FP64 master cache and materialize
BF16 only at the decoder boundary. Bolmo is therefore a candidate for a future
alias/identifier adapter, while the FP64-master policy remains the numerical
correctness mechanism.

## Actionable integration

- retain original UTF-8/byte spans in `OriginKey` evidence;
- train alias routing on byte-preserved forms, including punctuation and
  mixed-language identifiers;
- compile those aliases into the existing Port Plane instead of treating each
  spelling as new knowledge;
- evaluate exact alias recall separately from semantic retrieval recall.

No Bolmo checkpoint or language-quality claim is imported into this repository.
