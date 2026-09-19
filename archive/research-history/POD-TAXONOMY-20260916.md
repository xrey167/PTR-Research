# Structured Pod taxonomy

Vela-Domain supplies a soft domain signal. It should help narrow the Pod pool,
but it must not identify a source or override lifecycle checks.

Each Pod gets structured metadata:

```text
PodDescriptor
├── pod_type: context | math | model | reasoning | retrieval
├── domain: business | law | economics | engineering | ...
├── tags: supplier, alias, lead_time, multi_hop, numeric, policy, ...
└── hard: metadata provenance/ACL/generation authority
```

`route_candidates` applies hard domain/type/tag filters first and uses Vela only
as an advisory score. This supports many specialized Pods, including reasoning
and model Pods, without turning a misclassified Vela domain label into an
authorization or provenance decision.

`research/generate_taxonomy_dataset.py` creates a balanced routing set with 500
examples: exactly 100 per Pod type, with 60 train, 20 validation, and 20 test
examples for every type. In dataset `...-003`, validation and test use held-out
templates and held-out entity phrases. The reproducible Vela GPU run (seed 3407)
reached `1.00` train accuracy, `0.72` Pod-type and `0.90` domain accuracy on both
held-out splits (`300/100/100` rows). This exposes a real type-routing gap despite
more data, especially between model/reasoning/retrieval Pods.

The joint Dragonfly run additionally predicts `semantic_role` (`FACT`,
`CALCULATION`, `RULE`, `RELATION`, `DOCUMENT`) and `intent` alongside the
prototype address. Held-out Dragonfly address top-1 is `0.80`; type, role, and
intent are `0.84`, and domain is `0.80`. These are advisory learned signals.
Generation, ACL, provenance, and revocation remain deterministic lifecycle
fields and are never delegated to neural heads.

The trained checkpoint is `runs/dragonfly-taxonomy-router-001.pt`. It includes
the shared Dragonfly prototype address plus type, domain, role, intent, and
multi-label tag heads. The latest held-out run reaches Dragonfly top-1 `0.84`
and tag accuracy `0.934`.
