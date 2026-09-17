# Pod network transport benchmark

Measured on `xrserver` with the real threaded TCP/JSON-lines adapter, HMAC
signatures and manifest validation enabled:

- 2,000 requests
- 8 concurrent workers
- 100% successful responses
- **1,821 QPS**
- p50 **2.154 ms**
- p95 **2.659 ms**
- p99 **2.991 ms**

The same adapter accepts an `ssl.SSLContext` on both server and client for mTLS
deployment. The benchmark used loopback plaintext to isolate transport overhead;
production remote links should use TLS plus the existing HMAC/manifest checks.

With eight independent client processes over the server LAN address, the same
2,000-request run achieved **1,606 QPS**, p50 **1.177 ms**, p95 **1.339 ms**,
p99 **1.505 ms**, with a **100% success rate**. This is the current
multi-process network baseline; it is not yet a multi-host or distributed
storage benchmark.
