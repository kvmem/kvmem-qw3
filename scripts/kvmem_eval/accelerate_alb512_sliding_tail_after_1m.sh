#!/usr/bin/env bash
# Use the second vLLM sequence slot after the 1M Sliding run finishes.
#
# The auxiliary worker writes to a private output root.  Only completed,
# reusable checkpoints are copied into the canonical 512K root.  The primary
# worker later restores those checkpoints and remains the sole writer of the
# canonical answer/eval/manifest JSONL files, avoiding concurrent append races.
set -euo pipefail

PY=${PY:-/home/chaidi/qw3/.venv/bin/python}
RUNNER=${RUNNER:-/home/chaidi/AgentLongBench-Long/script/slidingWindow/run_sliding_window.py}
DATA512=${DATA512:-/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl}
ROOT512=${ROOT512:-/data/chaidi/kvmem_eval/results/agentlongbench_512k_sliding_window_cap64k_vllm022_fp8_mtp4_20260812}
ROOT1M=${ROOT1M:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_sliding_window_cap100k_vllm022_fp8_mtp4_20260812}
AUX=${AUX:-/data/chaidi/kvmem_eval/results/agentlongbench_512k_sliding_window_cap64k_vllm022_fp8_mtp4_20260812_tail80_aux}
LOG=${LOG:-/data/chaidi/kvmem_eval/logs/agentlongbench_512k_sliding_tail80_aux_20260812.log}
API_BASE=${API_BASE:-http://127.0.0.1:18123/v1}
MODEL=${MODEL:-Qwen3.6-27B-FP8}
TOKENIZER=${TOKENIZER:-/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8}

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

count_answers() {
  local path=$1
  if [[ -f "$path" ]]; then
    awk 'NF { n += 1 } END { print n + 0 }' "$path"
  else
    echo 0
  fi
}

answers_1m="$ROOT1M/answers/sliding_window.answers.jsonl"
echo "[$(date --iso-8601=seconds)] waiting for ALB1M Sliding 50/50"
while [[ "$(count_answers "$answers_1m")" -lt 50 ]]; do
  if ! curl --noproxy '*' -fsS --max-time 3 \
      "${API_BASE%/v1}/health" >/dev/null; then
    echo "[$(date --iso-8601=seconds)] serving endpoint is unavailable; exiting"
    exit 2
  fi
  sleep 30
done

echo "[$(date --iso-8601=seconds)] starting private 512K tail [80,100]"
set +e
env PYTHONUNBUFFERED=1 "$PY" "$RUNNER" run \
  --dataset-path "$DATA512" --output-root "$AUX" \
  --start-index 80 --limit 21 --window-prompt-tokens 65536 \
  --api-base "$API_BASE" --model "$MODEL" --api-backend vllm \
  --tokenizer-path "$TOKENIZER" --temperature 0.6 --top-p 0.95 --top-k 20 \
  --thinking-budget 8192 --answer-max-tokens 32768 \
  --chat-template-reserve 16 --seed 20260722 --continue-on-error \
  --timeout-sec 14400 --max-call-sec 14400 --attempts 2
status=$?
set -e

mkdir -p "$ROOT512/checkpoints"
copied=0
for checkpoint in "$AUX"/checkpoints/*.json; do
  [[ -f "$checkpoint" ]] || continue
  if "$PY" - "$checkpoint" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
final = value.get("final") or {}
raise SystemExit(0 if final.get("answer") and final.get("answer_finish_reason") != "length" else 1)
PY
  then
    target="$ROOT512/checkpoints/$(basename "$checkpoint")"
    if [[ ! -e "$target" ]]; then
      temporary="${target}.aux-copy.$$"
      cp "$checkpoint" "$temporary"
      # Publish only a complete JSON file.  If the canonical worker happened
      # to finish the same sample meanwhile, keep its checkpoint instead.
      if mv -n "$temporary" "$target"; then
        if [[ -e "$temporary" ]]; then
          rm -f "$temporary"
        else
          copied=$((copied + 1))
        fi
      fi
    fi
  fi
done

echo "[$(date --iso-8601=seconds)] tail worker status=$status copied=$copied"
# Individual failures are intentionally left for the canonical worker to run.
exit 0
