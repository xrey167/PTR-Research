#!/usr/bin/env bash
# Heterogeneous ensemble: GPU0 = qwen3b with reader-gen5 (+gen3 fallback pod),
# GPU1 = NeoHorse-1-4B with the Gen-6 reader adapter. stop kills both.
set -euo pipefail

QWEN=/home/xrey/neural-pods/models/qwen3b
NEOHORSE=/srv/ai/models/NeoHorse-1-4B
GEN3=/home/xrey/neural-pods/runs/qwen3b-lora-generation3-20260917/adapter
GEN5=/home/xrey/neural-pods/runs/qwen3b-lora-generation5-20260919/adapter
GEN6=/home/xrey/neural-pods/runs/neohorse-lora-generation6-20260919/adapter
LOG_DIR=/tmp/neural-pods-vllm-hetero
VLLM=/srv/ai/workspaces/vllm-venv/bin/vllm

stop_all() {
  fuser -k 18000/tcp 2>/dev/null || true
  fuser -k 18001/tcp 2>/dev/null || true
  for pid in $(pgrep -f "[v]llm serve" || true); do kill -TERM -- "-$pid" 2>/dev/null || true; kill -TERM "$pid" 2>/dev/null || true; done
  pgrep -f '[V]LLM::EngineCore' | xargs -r kill -TERM 2>/dev/null || true
  sleep 2
  pgrep -f '[V]LLM::EngineCore' | xargs -r kill -KILL 2>/dev/null || true
}

if [[ "${1:-start}" == "stop" ]]; then stop_all; echo '{"stopped":true}'; exit 0; fi

mkdir -p "$LOG_DIR"
stop_all

CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  nohup /srv/ai/workspaces/vllm-venv/bin/python "$VLLM" serve "$QWEN" \
  --served-model-name reader-base --enable-lora \
  --lora-modules "reader-gen3=$GEN3" "reader-gen5=$GEN5" \
  --max-lora-rank 32 --max-cpu-loras 2 \
  --dtype half --max-model-len 4096 --gpu-memory-utilization 0.80 --enforce-eager --port 18000 \
  >"$LOG_DIR/gpu0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  nohup /srv/ai/workspaces/vllm-venv/bin/python "$VLLM" serve "$NEOHORSE" \
  --served-model-name neohorse-base --enable-lora \
  --lora-modules "reader-gen6=$GEN6" \
  --max-lora-rank 32 --max-cpu-loras 1 \
  --dtype half --max-model-len 4096 --gpu-memory-utilization 0.80 --enforce-eager --port 18001 \
  >"$LOG_DIR/gpu1.log" 2>&1 &

for attempt in $(seq 1 240); do
  if curl -fsS --max-time 1 http://127.0.0.1:18000/health >/dev/null 2>&1 \
     && curl -fsS --max-time 1 http://127.0.0.1:18001/health >/dev/null 2>&1; then
    echo '{"ready":true,"gpu0":"qwen3b+gen3+gen5","gpu1":"neohorse+gen6"}'
    exit 0
  fi
  sleep 1
done
echo '{"ready":false,"error":"heterogeneous replicas did not become healthy"}' >&2
exit 1
