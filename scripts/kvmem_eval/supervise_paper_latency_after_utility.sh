#!/usr/bin/env bash
# Start isolated current-machine latency sampling after the queued utility and
# sliding-window runs have released the GPU.
set -euo pipefail

LOG=${LOG:-/data/chaidi/kvmem_eval/logs/paper_latency_after_utility_20260812.log}
ROOT=${ROOT:-/home/chaidi/qw3}
UTILITY_MARKER=${UTILITY_MARKER:-/data/chaidi/kvmem_eval/results/paper_utility_sliding_complete_20260812.json}
SUCCESS_MARKER=${SUCCESS_MARKER:-/data/chaidi/kvmem_eval/results/paper_latency_vllm_complete_20260812.json}

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for $UTILITY_MARKER"
while [[ ! -s "$UTILITY_MARKER" ]]; do
  sleep 30
done

echo "[$(date --iso-8601=seconds)] utility/sliding queue ended; checking GPU owners"
while pgrep -f '/bin/vllm serve .*Qwen3.6-27B-FP8' >/dev/null 2>&1; do
  sleep 30
done

echo "[$(date --iso-8601=seconds)] starting isolated vLLM latency sampling"
until "$ROOT/scripts/kvmem_eval/supervise_paper_latency_vllm_current_machine.sh"; do
  echo "[$(date --iso-8601=seconds)] vLLM latency pass interrupted; resuming in 30s"
  sleep 30
done
python3 "$ROOT/scripts/kvmem_eval/validate_paper_latency_current.py" \
  --root /data/chaidi/kvmem_eval/results/paper_pre_answer_latency_current_20260812 \
  --stage vllm --output "$SUCCESS_MARKER"
