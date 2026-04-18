#!/usr/bin/env bash
# Stop both llama-server instances started by start_servers.sh.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO/logs"

stop_one() {
  local name="$1" pidfile="$2"
  if [[ ! -f "$pidfile" ]]; then
    echo "[$name] no pidfile"
    return 0
  fi
  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[$name] stopping pid $pid (TERM)"
    kill -TERM "$pid" 2>/dev/null || true
    # give it a bit, then KILL if still up
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -0 "$pid" 2>/dev/null && { echo "[$name] sending KILL"; kill -KILL "$pid" 2>/dev/null || true; }
  else
    echo "[$name] not running"
  fi
  rm -f "$pidfile"
}

stop_one "model_a" "$LOG_DIR/llama-a.pid"
stop_one "model_b" "$LOG_DIR/llama-b.pid"

# also kill any stray llama-server processes inside the toolbox as a safety net
toolbox run --container llama-rocm-7.2 bash -c 'pkill -f llama-server 2>/dev/null || true' || true
echo "Done."
