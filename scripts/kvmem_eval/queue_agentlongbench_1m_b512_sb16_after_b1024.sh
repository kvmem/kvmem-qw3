#!/usr/bin/env bash
# Wait for the active B1024/SB32 full50 launcher to release port 8088, then
# start the matched B512/SB16 full50 experiment.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CURRENT_PID_FILE=/data/chaidi/kvmem_eval/logs/agentlongbench_1m_full50_k224k_g32k_b1024_sb32max_fp8_qrfix_immutable_refresh1_t06_think8k_20260727.pid
NEXT="$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_full50_k224k_b512_sb16.sh"

while [[ -e "$CURRENT_PID_FILE" ]]; do
  current_pid=$(<"$CURRENT_PID_FILE")
  if [[ -n "$current_pid" ]] && kill -0 "$current_pid" 2>/dev/null; then
    sleep 30
    continue
  fi
  break
done

while curl -fsS --noproxy '*' http://127.0.0.1:8088/health >/dev/null 2>&1; do
  sleep 5
done

exec "$NEXT"
