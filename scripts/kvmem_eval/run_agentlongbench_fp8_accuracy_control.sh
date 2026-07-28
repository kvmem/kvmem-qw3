#!/usr/bin/env bash
# Controlled AgentLongBench comparison for KV dtype and prefill chunk size.
#
# The four cells use the same current binary, prompt builder, selected sample
# IDs, decoding parameters, and KVMem configuration.  Every cell also writes a
# QW3_KVMEM_DUMP_SCORES file so output differences can be separated into
# retrieval-set changes and post-retrieval numerical changes.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE="$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh"

PORT=${PORT:-8088}
CPU_GB=${CPU_GB:-64}
EXPECTED=${EXPECTED:-}
QUESTION_IDS=${QUESTION_IDS:-}
TAG_PREFIX=${TAG_PREFIX:-agentlongbench_512k_fp8_accuracy_control_20260725}
WAIT_SESSION=${WAIT_SESSION:-}
WAIT_VALIDATION=${WAIT_VALIDATION:-}
CELLS=${CELLS:-fp16:2048,fp16:8192,fp8:2048,fp8:8192}

if [[ -z "$QUESTION_IDS" ]]; then
  echo "QUESTION_IDS must be a comma-separated stable-sample-ID list" >&2
  exit 2
fi

IFS=',' read -r -a ids <<<"$QUESTION_IDS"
if [[ -z "$EXPECTED" ]]; then
  EXPECTED=${#ids[@]}
fi
if [[ "$EXPECTED" -ne "${#ids[@]}" ]]; then
  echo "EXPECTED=$EXPECTED does not match ${#ids[@]} QUESTION_IDS" >&2
  exit 2
fi

if [[ -n "$WAIT_SESSION" ]]; then
  echo "[$(date --iso-8601=seconds)] waiting for tmux session $WAIT_SESSION"
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 30
  done
fi
if [[ -n "$WAIT_VALIDATION" ]]; then
  python3 - "$WAIT_VALIDATION" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("passed"):
    raise SystemExit(f"prerequisite validation failed: {report}")
print(f"prerequisite validation passed: {sys.argv[1]}", flush=True)
PY
fi

run_cell() {
  local dtype=$1
  local chunk=$2
  local cell="${dtype}_c${chunk}"
  local tag="${TAG_PREFIX}_${cell}"
  local result="/data/chaidi/kvmem_eval/results/$tag"
  mkdir -p "$result"
  echo "[$(date --iso-8601=seconds)] start $cell"
  env \
    PORT="$PORT" \
    TAG="$tag" \
    KV_DTYPE="$dtype" \
    PREFILL_CHUNK="$chunk" \
    KVMEM_OPT_LEVEL=opt_3 \
    CPU_GB="$CPU_GB" \
    NVME_GB=0 \
    QUESTION_IDS="$QUESTION_IDS" \
    EXPECTED="$EXPECTED" \
    METHOD="kvmem_control_${dtype}_c${chunk}_current" \
    QW3_KVMEM_DUMP_SCORES="$result/kvmem_scores.jsonl" \
    "$BASE"
  echo "[$(date --iso-8601=seconds)] complete $cell"
}

IFS=',' read -r -a cells <<<"$CELLS"
for cell in "${cells[@]}"; do
  if [[ ! "$cell" =~ ^(fp16|fp8):([1-9][0-9]*)$ ]]; then
    echo "invalid CELLS entry: $cell (expected fp16:CHUNK or fp8:CHUNK)" >&2
    exit 2
  fi
  run_cell "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
done

echo "[$(date --iso-8601=seconds)] requested accuracy-control cells completed: $CELLS"
