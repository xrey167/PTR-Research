#!/usr/bin/env bash
set -euo pipefail

# vLLM 0.29 on this host uses CUDA 13 wheels while the system nvcc is CUDA 12.8.
# Disable the optional FlashInfer sampler JIT; vLLM falls back to its native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/srv/ai/models/NeoHorse-1-4B}"
PORT="${PORT:-18000}"

exec /srv/ai/workspaces/vllm-venv/bin/vllm serve "$MODEL_PATH" \
  --served-model-name neohorse-1-4b \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype half \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --enforce-eager
