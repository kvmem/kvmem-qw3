#!/usr/bin/env bash
# Wait for the active four-sample oracle-only run to finish, then launch the
# matched message-expansion/length-normalized-mass experiment on port 8088.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ORACLE_ROOT=${ORACLE_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_round_padding_oracle4_only_20260728_230124}
ORACLE_EVAL=${ORACLE_EVAL:-$ORACLE_ROOT/eval.jsonl}
PORT=${PORT:-8088}
POLL_SECONDS=${POLL_SECONDS:-30}

echo "[queue] waiting for four completed oracle eval rows: $ORACLE_EVAL"
while true; do
  completed=0
  if [[ -f "$ORACLE_EVAL" ]]; then
    completed=$(sed '/^[[:space:]]*$/d' "$ORACLE_EVAL" | wc -l)
  fi
  if (( completed >= 4 )); then
    echo "[queue] oracle complete: $completed/4"
    break
  fi
  if ! pgrep -f \
      'run_agentlongbench_kvmem_oracle.py.*agentlongbench_1m_round_padding_oracle4_only_20260728_230124' \
      >/dev/null 2>&1; then
    echo "[queue] oracle runner exited before producing four eval rows" >&2
    exit 6
  fi
  echo "[queue] oracle progress: $completed/4"
  sleep "$POLL_SECONDS"
done

# The oracle launcher owns its server and removes it through its EXIT trap.
for _ in $(seq 1 120); do
  if ! curl -fsS --noproxy '*' \
      "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if curl -fsS --noproxy '*' \
    "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[queue] port $PORT is still occupied after oracle completion" >&2
  exit 7
fi

echo "[queue] starting message mass experiment"
exec env PORT="$PORT" \
  bash "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_message_mass_oracle4.sh"
