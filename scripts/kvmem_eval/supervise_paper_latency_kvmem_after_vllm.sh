#!/usr/bin/env bash
set -euo pipefail

LOG=${LOG:-/data/chaidi/kvmem_eval/logs/paper_latency_kvmem_after_vllm_20260812.log}
ROOT=${ROOT:-/home/chaidi/qw3}
OUT=${OUT:-/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_current_20260812}
VLLM_MARKER=${VLLM_MARKER:-/data/chaidi/kvmem_eval/results/paper_latency_vllm_complete_20260812.json}
SUCCESS_MARKER=${SUCCESS_MARKER:-/data/chaidi/kvmem_eval/results/paper_latency_kvmem_complete_20260812.json}

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for $VLLM_MARKER"
while [[ ! -s "$VLLM_MARKER" ]]; do sleep 30; done
while pgrep -f '/bin/vllm serve .*Qwen3.6-27B-FP8' >/dev/null 2>&1; do sleep 30; done
echo "[$(date --iso-8601=seconds)] starting KVMem latency"
until "$ROOT/scripts/kvmem_eval/supervise_paper_latency_kvmem_long_current_machine.sh"; do
  echo "[$(date --iso-8601=seconds)] KVMem latency pass interrupted; resuming in 30s"
  sleep 30
done
python3 "$ROOT/scripts/kvmem_eval/validate_paper_latency_current.py" \
  --root "$OUT" --stage all --output "$OUT/final_all_validation.json"
python3 "$ROOT/scripts/kvmem_eval/summarize_paper_pre_answer_latency_current.py" \
  --root "$OUT" --json-output "$OUT/summary.json" \
  --markdown-output "$OUT/summary.md" \
  --latex-output "$OUT/latency_rows.tex"
# The public completion marker is written last, only after both validation and
# aggregation succeed.
python3 "$ROOT/scripts/kvmem_eval/validate_paper_latency_current.py" \
  --root "$OUT" --stage all --output "$SUCCESS_MARKER"
