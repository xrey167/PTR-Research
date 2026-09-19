# Real model probe — 2026-09-17

Checkpoint: `/srv/ai/workspaces/models/NeoHorse-1-4B` on `xrserver`, CUDA,
Transformers generation, three prompts, 32 new tokens each.

| Metric | Result |
|---|---:|
| Generated tokens | 96 |
| Total elapsed | 6.09 s |
| Mean generation speed | 15.77 tokens/s |
| Device | CUDA (`cuda:0`) |
| Precision | float16 |

The probe uses the model's native generation path and is an engineering
throughput check, not the official NeoHorse ten-benchmark protocol. The log
reported fallback PyTorch implementations for `causal_conv1d` and
`flash-linear-attention`; installing compatible CUDA kernels is a separate
optimization gate and should be measured against this baseline.

The attempted compatible installation was stopped before changing the
environment because the server image has no `nvcc` compiler. A source build
therefore cannot produce the CUDA extension; the current result is the
reproducible fallback baseline.

With one independent model process per RTX 3090 and 16 generated tokens per
prompt, the initial measured per-GPU rates were 17.90 and 14.85 tokens/s, or
about 32.75 tokens/s aggregate while both GPUs were active.

The server now has the CUDA 12.4 development toolkit and a locally compiled
`causal-conv1d` 1.7.0 extension. A repeatable single-GPU probe improved the
32-token mean from 15.77 to **22.57 tokens/s** (+43%). Installing
`flash-linear-attention` 0.5.2 is successful; after Triton warm-up its
repeatable mean was **22.02 tokens/s**, so it is retained for compatibility but
does not improve this short generation workload over causal-conv1d alone.

Running one optimized process per GPU with 16 generated tokens measured
**22.39** and **18.38 tokens/s**, or **40.77 tokens/s aggregate** across both
RTX 3090 cards.

## Batched generation throughput

The same optimized Transformers path was measured with a warmup generation
followed by 32 new tokens. These figures measure serving fan-out efficiency;
they are not a claim about the official NeoHorse benchmark suite.

| GPU | Batch | Tokens/s | Sequences/s |
|---|---:|---:|---:|
| RTX 3090 #0 | 1 | 35.81 | 1.12 |
| RTX 3090 #0 | 2 | 71.66 | 2.24 |
| RTX 3090 #0 | 4 | 137.15 | 4.29 |
| RTX 3090 #1 | 1 | 35.19 | 1.10 |
| RTX 3090 #1 | 2 | 67.95 | 2.12 |
| RTX 3090 #1 | 4 | 156.71 | 4.90 |

At batch four the two cards reached **293.86 generated tokens/s** in separate
processes. Relative to batch one on the same cards this is a 4.14x throughput
increase for a 4x larger batch, which is the expected operating point for
high-throughput pod/router fan-out. The result is a short-window measurement;
long-context serving, admission control, KV-cache pressure, and sustained
multi-minute load still need separate tests.

## Sustained batch load

A repeated 20-iteration run (two warmup generations, batch 4, 32 new tokens per sequence) remained stable:

| GPU | Tokens/s | p50 latency | p95 latency | Max allocated |
|---|---:|---:|---:|---:|
| RTX 3090 #0 | 154.14 | 829.6 ms | 836.7 ms | 8,245 MiB |
| RTX 3090 #1 | 147.34 | 859.9 ms | 946.9 ms | 8,245 MiB |
| Both GPUs | **301.48 tok/s** | separate processes | separate processes | 16,490 MiB total |

This confirms that the short batch result is not only a one-shot warmup artifact. It is still an inference throughput test, not the official NeoHorse benchmark suite or a long-context serving test.

## Batch-size sweep on one RTX 3090

The memory headroom allows much larger fan-out batches. These are short
stability probes (10 iterations for batches 8–16, 5 for 32, 3 for 64, 2 for
128, and 1 for 256), each with 32 generated tokens:

| Batch | Tokens/s | p50 latency | Max allocated |
|---:|---:|---:|---:|
| 8 | 305.92 | 828 ms | 8,462 MiB |
| 16 | 573.31 | 892 ms | 8,898 MiB |
| 32 | 921.01 | 1,111 ms | 9,760 MiB |
| 64 | 1,243.86 | 1,645 ms | 11,484 MiB |
| 128 | 1,417.48 | 2,890 ms | 14,934 MiB |
| 256 | 1,510.59 | 5,423 ms | 21,837 MiB |

Batch 64–128 is the practical operating range: it keeps latency bounded while
using the available GPU efficiently. Batch 256 approaches the 24 GiB card
limit and is a throughput stress point rather than a safe default.

## Full Pod transport with NeoHorse

256 real PodRequests through PodFanout and BatchedPodHandler, batch limit 64, 32 generated tokens: GPU0 778.29 tok/s (24.32 req/s), GPU1 856.39 tok/s (26.76 req/s), aggregate **1,634.68 tok/s**. All 512 responses were correct with zero transport/model errors, rejects or deadline drops; max allocated memory was 10,616 MiB per card.

## Concurrent two-GPU Pod load

Both model-backed Pod handlers ran simultaneously (256 requests each, batch 64, 32 generated tokens): GPU0 756.51 tok/s, GPU1 715.93 tok/s, **1,472.43 tok/s aggregate**, 512/512 correct responses, zero errors/rejects/deadline drops. Peak allocations were 10,508 and 11,206 MiB.

## TCP socket model path

A NeoHorse handler behind PodSocketServer processed 128 real TCP-framed PodRequests with 32 concurrent clients: **282.03 tok/s**, 8.81 requests/s, 128/128 correct, zero errors. This includes socket setup per request, JSON framing, protocol validation, adaptive batching and GPU generation; persistent sessions remain the next transport optimization.

Persistent TCP sessions improved the same socket model test to **452.13 tok/s** (128 requests, 32 workers, 128/128 correct, zero errors), versus 282.03 tok/s with one connection per request: **1.60x faster**.

## Windows ? xrserver ? GPU model

A Windows client sent 64 real requests over the LAN to a NeoHorse PodSocketServer bound on xrserver, using 16 persistent TCP sessions: **250.92 tok/s**, 7.84 requests/s, 64/64 correct, zero errors. This includes cross-host network latency and remote GPU generation.

Windows ? xrserver mTLS model path

A Windows client sent 64 real NeoHorse PodRequests over the LAN using 16 persistent sessions with mutual TLS, HMAC request authentication, and manifest-hash validation: **253.96 tok/s**, 7.94 requests/s, 64/64 correct, zero errors. Temporary test certificates were used; production rotation is still open.

Full integrated model path: Windows ? mTLS + HMAC + manifest validation ? resource admission lease ? adaptive batching ? NeoHorse-1-4B on xrserver. 64 requests / 16 persistent sessions produced 13.757 req/s and 220.112 tok/s, 64/64 correct and 0 errors; GPU0 peaked at 9,216 MiB and resources were released on shutdown.

Admission rejection test: with an intentionally impossible 100 GiB per-request budget, 8/8 real mTLS requests were rejected before inference (0.2325 s, zero model work). GPUs returned to 1 MiB after shutdown.

Model residency integration: NeoHorse was loaded only after a 10 GiB generation-bound residency lease was acquired. With 256 MiB request leases, mTLS/HMAC and batching, 32/32 Windows requests were correct at 11.533 req/s and 92.263 tok/s; server tracker ended with 32 completed, 0 errors, 0 active, p50 510.652 ms / p95 1,283.266 ms. GPUs returned to 1 MiB on shutdown.
