# Granular Pod Schema and Stable Branching

Pods now have two identifiers with different jobs:

- `pod_identity` is stable for the composed Pod contract (type, name, role,
  domain and contract). Model links can retain this identity across updates.
- `artifact_key` is immutable and identifies one concrete revision, including
  its lineage and bound knowledge generation.

`PodMetadata` makes the composition surface explicit: semantic role, domain,
tags, aliases, capabilities, entities, sensitivity, confidence, ACL, temporal
scope, retention and cluster key. `PodBuilder` turns these fields plus a
type-specific contract into a validated registry artifact. `branch()` creates
a new immutable revision while preserving `pod_identity`; the old revision is
automatically stale when a new knowledge generation becomes the registry head.

`resolve_pod_identity()` resolves the active revision at the lifecycle
boundary. Revocation, ACL and generation checks still happen in the registry,
so a stable model link never bypasses lifecycle authorization.

Validation: `tests/test_pod_builder.py`; full suite: **219 passed**.
