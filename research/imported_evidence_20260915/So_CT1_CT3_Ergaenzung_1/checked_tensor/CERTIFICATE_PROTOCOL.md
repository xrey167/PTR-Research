# CT2: exact modular product checks and operation-count gate

Registered after CT1; separate mathematical experiment, no model execution.
Question: is a Freivalds-style product check cheaper for the actual TC1 rank?
Known algorithm, no novelty claim. Test over exact dyadic inputs from CT1.

Scale tensors by 16 to integers; the output by 256. Check
16*(B+S+R) + C*(Y-X) = O over prime 2^31-1, with strict input/output magnitude
bounds making equality modulo p imply equality over integers in this domain.
Use 40 independent binary challenge vectors per candidate. In production they
must be sampled secretly AFTER the producer commits its output. Reproducible
test seeds are public; tests do not provide adversarial soundness.

Before evaluation: 36 valid CT1 outputs accepted; one-entry errors rejected;
addition of the modulus rejected by the magnitude check. CT1 independent
Fraction reference remains the deterministic strong correctness control.
Probability bound <=2^-40 for a fixed nonzero error matrix is theoretical and
must never be reported as an empirically measured failure probability.

Count scalar multiply-accumulate terms only, not hardware speed:
direct m*k*d; verifier 40*(k*d+m*k+m*d), ignoring common costs in both.
Gate: certificate cheaper for TC1 local matrix shape m=15,k=6,d=256, and
single-pair shape m=1,k=6,d=256. Failure excludes this certificate as a
cost-saving replacement for TC1, regardless of correctness. Include a large
512-cube positive-cost control without claiming it occurs in this project.
