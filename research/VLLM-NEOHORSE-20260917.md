# vLLM NeoHorse validation — 2026-09-17

## Environment

- Host: `xrserver`, RTX 3090, CUDA driver 595.91.07
- vLLM: `0.29.0` in `/srv/ai/workspaces/vllm-venv`
- Checkpoint: `/srv/ai/models/NeoHorse-1-4B`
- Existing training environment was not modified.

## Compatibility finding

The bundled FlashInfer sampler attempted to compile with `nvcc --compress-mode=size`,
which is not accepted by the host CUDA 12.8 compiler. The server failed before
opening its HTTP port. Setting `VLLM_USE_FLASHINFER_SAMPLER=0` selects vLLM's native
sampler and removes that JIT dependency. `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the
single-GPU validation deterministic.

## Measured result

OpenAI-compatible `/v1/chat/completions`, `max_tokens=8`, deterministic temperature 0:

| workers | requests | success | errors | requests/s | tokens/s | p50 | p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 32 | 0 | 4.75 | 37.98 | 205 ms | 231 ms |
| 8 | 64 | 64 | 0 | 31.35 | 250.80 | 243 ms | 338 ms |

The run is stored in `research/runs/neohorse-vllm-20260917.jsonl`.

The existing PodSocket path measured 116.86 tokens/s at the same output length;
vLLM reached 250.80 tokens/s under concurrent load (2.15× higher), while preserving
the same checkpoint and GPU. After shutdown both GPUs returned to 1 MiB usage.

The replica router (`neural_pods.vllm_router`) was then tested against two live
vLLM instances: 128/128 requests succeeded at 369.47 tokens/s with exactly 64
requests per GPU. After GPU 0 was stopped deliberately, 32/32 requests still
succeeded through GPU 1 and the router recorded 16 bounded failovers.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 ./research/run_vllm_neohorse.sh
```
