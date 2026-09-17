#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/srv/ai/models/NeoHorse-1-4B}"
LOG_DIR="${LOG_DIR:-/tmp/neural-pods-vllm}"
PORT0="${PORT0:-18000}"
PORT1="${PORT1:-18001}"

stop_replica() {
  local port="$1"
  local api_pids
  api_pids="$(pgrep -f "[v]llm serve .*--port ${port}" || true)"
  fuser -k "${port}/tcp" 2>/dev/null || true
  # vLLM's engine children share the API process group when launched via setsid.
  for pid in $api_pids; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  # Also reap orphaned engine processes from a prior interrupted launch.
  pgrep -f '[V]LLM::EngineCore' | xargs -r kill -TERM 2>/dev/null || true
  pgrep -f 'vllm-venv/bin/python3.12 -c from multiprocessing.resource_tracker' | xargs -r kill -TERM 2>/dev/null || true
  sleep 1
  pgrep -f '[V]LLM::EngineCore' | xargs -r kill -KILL 2>/dev/null || true
  pgrep -f 'vllm-venv/bin/python3.12 -c from multiprocessing.resource_tracker' | xargs -r kill -KILL 2>/dev/null || true
}

stop_all() {
  stop_replica "$PORT0"
  stop_replica "$PORT1"
}

if [[ "${1:-start}" == "stop" ]]; then
  stop_all
  echo '{"stopped":true}'
  exit 0
fi

mkdir -p "$LOG_DIR"
stop_all
CUDA_VISIBLE_DEVICES=0 PORT="$PORT0" MODEL_PATH="$MODEL_PATH" \
  setsid env CUDA_VISIBLE_DEVICES=0 PORT="$PORT0" MODEL_PATH="$MODEL_PATH" \
  "$ROOT/run_vllm_neohorse.sh" >"$LOG_DIR/gpu0.log" 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 PORT="$PORT1" MODEL_PATH="$MODEL_PATH" \
  setsid env CUDA_VISIBLE_DEVICES=1 PORT="$PORT1" MODEL_PATH="$MODEL_PATH" \
  "$ROOT/run_vllm_neohorse.sh" >"$LOG_DIR/gpu1.log" 2>&1 < /dev/null &

for attempt in $(seq 1 120); do
  if curl -fsS --max-time 1 "http://127.0.0.1:${PORT0}/health" >/dev/null 2>&1 \
     && curl -fsS --max-time 1 "http://127.0.0.1:${PORT1}/health" >/dev/null 2>&1; then
    printf '{"ready":true,"ports":[%s,%s],"model":"%s"}\n' "$PORT0" "$PORT1" "$MODEL_PATH"
    exit 0
  fi
  sleep 1
done
echo '{"ready":false,"error":"vLLM replicas did not become healthy"}' >&2
exit 1
