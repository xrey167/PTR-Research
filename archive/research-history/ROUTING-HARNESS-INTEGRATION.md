# Routing-harness integration

The project now has a deterministic `RoutingHarness` for the NeoHorse-style loop: estimate task capability demand, select a candidate from a heterogeneous Model-Pod pool, record tool calls and outcomes, and aggregate only verified capability feedback for the next training mixture.

Selection uses capability overlap, an optional latency budget, and a small cost term. Candidate identity, model family, and artifact key stay explicit; this layer does not bypass registry generation or revocation checks.

The implementation is `neural_pods/routing_harness.py`, with tests in `tests/test_routing_harness.py`.
