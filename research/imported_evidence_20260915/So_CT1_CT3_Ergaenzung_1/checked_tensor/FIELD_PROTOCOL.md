# CT3 — adaptive development, not independent semantic validation

After CT2's 40-binary-challenge cost failure, evaluate the standard alternative
of two uniformly sampled field challenges over p=2^31-1. For a fixed nonzero
error matrix, the theoretical false-accept bound becomes <=p^-2<2^-61.
This is a known Freivalds variant, not an invention. CT1 dyadic inputs are
reused for arithmetic checks; no fresh domain or model validation is claimed.

Check all 36 stored cases against exact products; reject every fixed one-entry
error. Random challenges are generated reproducibly for audit, NOT secrets for
adversarial deployment. Use the enforced closed shapes/magnitudes of CT2 to
bound int64 intermediates. Reject a modular alias before arithmetic.

Cost comparison: 2*(k*d+m*k+m*d) versus m*k*d. Report m=1, m=3 (maximum active
pair terms in one original three-source query), m=15 (entire pair bank), k=6,
d=256. Hypothesis passes on-query savings only if m=3 is cheaper. Do not
substitute an all-15-pair installation win for a single-query inference win.
