# Cross-host Pod transport benchmark

The Pod server ran on `xrserver` at `192.168.1.223:39123`; the client ran from
the Windows workstation over the LAN. The server was stopped after the test.

- 1,000 requests over one persistent TCP session
- HMAC and manifest verification active
- success rate: **100%**
- **1,796.94 QPS**
- p50: **0.527 ms**
- p95: **0.713 ms**
- p99: **0.889 ms**

This proves the Pod protocol works across the host boundary. The test is still
single-server (the Windows machine is the client); it does not claim replicated
storage or multi-server consensus.
