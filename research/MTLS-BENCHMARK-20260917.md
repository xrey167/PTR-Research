# Mutual-TLS transport benchmark

Measured on `xrserver` with temporary CA, server and client certificates,
mutual certificate verification, HMAC and manifest verification enabled:

- 100 requests
- success rate: **100%**
- p50: **0.102 ms**
- p95: **0.151 ms**
- p99: **0.264 ms**

The benchmark reuses one authenticated TLS connection (the production path).
Opening a new connection per request measured roughly 43 ms p50 due to the TLS
handshake, which is why the persistent-session API is the required hot path.
Test certificates were temporary and removed after the run.
