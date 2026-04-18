#!/usr/bin/env bash
# Starts two llama.cpp servers on the HOST (not the toolbox), using the
# host-built llama-server at ~/Work/amd-strix-halo-toolboxes/llama.cpp/build/bin
# which is recent enough to include gemma4 support and was linked against the
# host's newer glibc. ROCm libs are taken from /opt/rocm.
#
# Each server uses --parallel for continuous batching and --cache-reuse to
# reuse the shared <system><rubric><task> KV prefix across slots.
#
# Logs -> logs/llama-a.log, logs/llama-b.log
# PIDs -> logs/llama-a.pid, logs/llama-b.pid

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="/home/marcel/Work/amd-strix-halo-toolboxes/models"
LOG_DIR="$REPO/logs"

LLAMA_BIN="/home/marcel/Work/amd-strix-halo-toolboxes/llama.cpp/build/bin/llama-server"
ROCM_LIB_DIR="/opt/rocm/lib"

MODEL_A_PATH="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_B_PATH="$MODELS_DIR/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

PORT_A=8080
PORT_B=8081

# Batch throughput knobs
PARALLEL=6        # continuous-batching slots per server
CACHE_REUSE=256   # reuse shared prefix chunks >=256 tokens across slots
CTX=49152         # total KV; per slot = CTX / PARALLEL ≈ 8192

mkdir -p "$LOG_DIR"

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "FATAL: llama-server not found or not executable at $LLAMA_BIN" >&2
  exit 1
fi

start_one() {
  local alias="$1" port="$2" model_path="$3" logfile="$4" pidfile="$5"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[$alias] already running (pid $(cat "$pidfile"))"
    return 0
  fi

  if [[ ! -f "$model_path" ]]; then
    echo "[$alias] FATAL: model not found at $model_path" >&2
    return 1
  fi

  echo "[$alias] starting on port $port, model=$(basename "$model_path")"
  echo "[$alias]   parallel=$PARALLEL cache_reuse=$CACHE_REUSE ctx=$CTX"

  nohup setsid env \
    LD_LIBRARY_PATH="$ROCM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    HIP_VISIBLE_DEVICES=0 \
    "$LLAMA_BIN" \
      -m "$model_path" \
      -c "$CTX" \
      --parallel "$PARALLEL" \
      --kv-unified \
      --cache-reuse "$CACHE_REUSE" \
      --slot-prompt-similarity 0.1 \
      -ngl 999 \
      -fa 1 \
      --no-mmap \
      --threads 8 \
      -b 4096 -ub 4096 \
      --prio 2 \
      --host 127.0.0.1 \
      --port "$port" \
      --alias "$alias" \
      --jinja \
      --reasoning-format deepseek \
      >"$logfile" 2>&1 &

  echo $! >"$pidfile"
  disown || true
  echo "[$alias] pid=$(cat "$pidfile")  log=$logfile"
}

start_one "model_a" "$PORT_A" "$MODEL_A_PATH" "$LOG_DIR/llama-a.log" "$LOG_DIR/llama-a.pid"
start_one "model_b" "$PORT_B" "$MODEL_B_PATH" "$LOG_DIR/llama-b.log" "$LOG_DIR/llama-b.pid"

echo
echo "Waiting for both servers to become ready (poll /health)..."
for port in "$PORT_A" "$PORT_B"; do
  ok=0
  for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "  port $port: ready (${i}s)"
      ok=1
      break
    fi
    sleep 1
  done
  if [[ "$ok" -eq 0 ]]; then
    echo "  port $port: NOT ready after 180s — see $LOG_DIR/llama-*.log"
  fi
done

echo "Done. Tail logs with:  tail -f $LOG_DIR/llama-*.log"
