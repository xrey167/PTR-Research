# Neural Pods architecture validation â€” 2026-09-17

The measurements below were run on `xrserver` (2Ã— RTX 3090, 24 GiB each)
and from the Windows client over the LAN.

## Gate

- 268 tests passed, 1 skipped, 15 warnings.
- LoRA A/B gate: held-out guarded exact match improved from **84/132** to
  **92/132** after a real CUDA training and reload cycle.
- LoRA A/B gate (dev split): same improvement confirmed independently
  (84/132 base vs **92/132** adapter).
- Retrieval recall@5: 1.0; HNSW recall@10: 1.0; MRR: 1.0.
- Cache hit rate: 98.5%; p99: 0.213 ms.
- Authenticated transport p95: 0.151 ms.
- Raft: 3 replicas; persistent TCP 9,909 proposals/s; mTLS 2,333 proposals/s.
- Quorum: 1,000/1,000 successful writes with one failed replica.
- PostgreSQL quorum adapter: 3 replicas / quorum 2, healthy write p95
  **2.17 ms**, latest revision read back correctly after one connection loss.
- Resource leases: zero active leases after every run.

## Model serving

| Path | Load | Result |
|---|---:|---:|
| Transformers PodSocket, one GPU | 6.54 s | 116.86 tokens/s |
| vLLM, one GPU, 8 workers | â€” | 250.80 tokens/s; 64/64 |
| vLLM, two replicas, loopback | â€” | 401.47 tokens/s; 128/128 |
| vLLM Router, Windows â†’ LAN | â€” | 370.38 tokens/s; 128/128 |
| vLLM Router, LAN with GPU 0 stopped | â€” | 32/32; 16 failovers; 0 errors |

Healthy LAN routing distributed exactly 64 requests to each GPU. The deliberate
GPU-0 failure was detected by health checks and all requests were retried once on
GPU 1 within the bounded retry policy.

## Scheduler and lifecycle

- Adaptive batcher: **9,283 requests/s**, 10,000/10,000 correct, 318 batches,
  mean batch size 31.45, maximum queue wait 47.56 ms,
  zero rejects, cancellations, deadline drops, or errors.
- Resource runtime: **29,590 requests/s** for 10,000 tracked admissions,
  10,000/10,000 successful, tracker p50 **0.652 ms**, p95 **1.372 ms**, and
  zero active records after completion.

## Reproduction

```bash
python research/verify_architecture_gate.py
python -m pytest -q
python research/benchmark_vllm_router_lan.py
PYTHONPATH=. python research/benchmark_adaptive_batcher.py --output research/runs/adaptive-batcher-20260917.json
PYTHONPATH=. python research/benchmark_postgres_quorum.py
```

The vLLM launcher is `research/run_vllm_neohorse.sh`. FlashInfer sampling is
disabled because its CUDA 13 JIT flag is incompatible with the host CUDA 12.8
`nvcc`; the native vLLM sampler is used instead.

For a reproducible two-GPU deployment, use
`research/run_vllm_replicas.sh start` and stop it with
`research/run_vllm_replicas.sh stop`. The stop path reaps API, engine-core and
resource-tracker children; the final validation left both GPUs at 1 MiB used.

The router accepts an SSL context for HTTPS/mTLS deployments and passes it to
both health and chat requests; this is covered by the router test suite.

