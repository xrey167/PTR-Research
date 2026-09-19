#!/usr/bin/env bash
# Start two vLLM replicas of the qwen3b base, each serving both reader
# adapters (reader-gen3, reader-gen5) via vLLM multi-LoRA. stop kills both.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/xrey/neural-pods/models/qwen3b}"
GEN3="${GEN3:-/home/xrey/neural-pods/runs/qwen3b-lora-generation3-20260917/adapter}"
GEN5="${GEN5:-/home/xrey/neural-pods/runs/qwen3b-lora-generation5-20260919/adapter}"
LOG_DIR="${LOG_DIR:-/tmp/neural-pods-vllm-ensemble}"
PORT0="${PORT0:-18000}"
PORT1="${PORT1:-18001}"
VLLM_PY=/srv/ai/workspaces/vllm-venv/bin/vllm

stop_replica() {
  local port="$1"
  fuser -k "${port}/tcp" 2>/dev/null || true
  local api_pids
  api_pids="$(pgrep -f "[v]llm serve .*--port ${port}" || true)"
  for pid in $api_pids; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  pgrep -f '[V]LLM::EngineCore' | xargs -r kill -TERM 2>/dev/null || true
  sleep 1
  pgrep -f '[V]LLM::EngineCore' | xargs -r kill -KILL 2>/dev/null || true
}

if [[ "${1:-start}" == "stop" ]]; then
  stop_replica "$PORT0"
  stop_replica "$PORT1"
  echo '{"stopped":true}'
  exit 0
fi

mkdir -p "$LOG_DIR"
stop_replica "$PORT0"
stop_replica "$PORT1"

launch() {
  local gpu="$1" port="$2"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0     nohup /srv/ai/workspaces/vllm-venv/bin/python "$VLLM_PY" \
    serve "$MODEL_PATH" \
    --served-model-name reader-base \
    --enable-lora \
    --lora-modules "reader-gen3=$GEN3" "reader-gen5=$GEN5" \
    --max-lora-rank 32 --max-cpu-loras 2 \
    --dtype half --max-model-len 4096 --gpu-memory-utilization 0.80 \
    --enforce-eager --port "$port" \
    >"$LOG_DIR/gpu$gpu.log" 2>&1 &
}

launch 0 "$PORT0"
launch 1 "$PORT1"

for attempt in $(seq 1 180); do
  if curl -fsS --max-time 1 "http://127.0.0.1:${PORT0}/health" >/dev/null 2>&1 \
     && curl -fsS --max-time 1 "http://127.0.0.1:${PORT1}/health" >/dev/null 2>&1; then
    echo '{"ready":true,"ports":['"$PORT0"','$PORT1'],"adapters":["reader-gen3","reader-gen5"]}'
    exit 0
  fi
  sleep 1
done
echo '{"ready":false,"error":"ensemble replicas did not become healthy"}' >&2
exit 1
