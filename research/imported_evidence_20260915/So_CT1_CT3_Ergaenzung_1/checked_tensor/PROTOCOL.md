# CT1: checked tensor expressions — preregistered mathematics experiment

Primary input archive v11 resolves, but two transfers returned HTTP 502.
This standalone work is based on the TC1 formula provided in the conversation.
No archived tensors, Qwen weights or fresh model outputs are available here.
Do not merge until the latest archive is materialized and verified.

Question: can the numeric build claim be checked against an authoritative
snapshot instead of trusting a free-form compiler's declared read set?

Candidate: bounded typed tensor expression data (input/add/sub/matmul only),
no executable Python or arbitrary filesystem/network operations in expressions.
Snapshot input identities bind tensor shapes, FP64 bytes and monotonically
increasing epochs. Expressions bind all transitive leaves; verification
recomputes the result under the current snapshot and binds program, arithmetic
ABI, input manifest and output digest. This is deterministic replay, NOT a
cryptographic proof, sandbox for arbitrary programs, or proof of model quality.

Expression under test: B + S + R + C @ (Y-X). The shared C is treated as an
input, NOT falsely claimed to be proven equal to R X^+. Its derivation and
canonical source revision must be bound separately in full integration.

Development seeds 11, 19, shape (2,4,2); source and protocol sealed after these
tests. Validation seeds 101,103,107,109,113,127,131,137,139,149,151,157 with
shapes (2,4,2),(3,8,3),(4,16,4): 36 cases. Matrices use bounded dyadic numbers
(integer/16), giving exactly representable sums/products in the tested range.
Independent Fraction arithmetic must equal every FP64 output element exactly.

For every case: fresh acceptance; stale snapshot rejection; refreshed snapshot
with the previous output rejected when a constructed change affects output;
undeclared leaf rejected; epoch-only ABA rejects old evidence and accepts fresh
evidence even if output bytes are unchanged. Tests distinguish snapshot
freshness, numeric correctness and semantic correctness.

Structural controls: unsupported operation, unknown leaf, malformed reference,
shape mismatch, nonfinite tensor and forged arithmetic ABI rejected. A bare
hash/manifest comparator intentionally accepts relabelled bytes; a conventional
independent full replay is the strong correctness control. No advantage over
ordinary build systems is inferred.

Gates: all 36 exact arithmetic cases and specified negative controls pass.
New source revision needed after any post-seal fix. Preserve failures. Measure
verification work separately; neither <5% lifecycle overhead nor >10x JIT can
be claimed from this small mathematical experiment.
