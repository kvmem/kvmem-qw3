#!/usr/bin/env bash
# Single-sample 2x2x2 factorial ablation of the three paper-facing KVMem
# performance optimizations. Every cell uses the same binary, data row, model
# configuration, storage budget, and deterministic decoding parameters.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER="$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh"
PORT=${PORT:-8088}
QUESTION_ID=${QUESTION_ID:-adb0765b3c59611b3b9923d5b06e6dfddf70021a16c0ea948c219c40085642e5}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
TAG_PREFIX=${TAG_PREFIX:-agentlongbench_512k_kvmem_core_ablation_${STAMP}}
TEMP=${TEMP:-0}
THINKING_BUDGET=${THINKING_BUDGET:-4096}
KV_DTYPE=${KV_DTYPE:-fp16}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
KVMEM_BUDGET=${KVMEM_BUDGET:-204800}
GEN_BUDGET=${GEN_BUDGET:-32768}
CPU_GB=${CPU_GB:-64}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
RESULTS_ROOT=${RESULTS_ROOT:-/data/chaidi/kvmem_eval/results}
SUMMARY_LOG="$LOG_ROOT/${TAG_PREFIX}_ablation.log"
SUMMARY_JSON="$RESULTS_ROOT/${TAG_PREFIX}_performance_summary.json"
SUMMARY_MD="$RESULTS_ROOT/${TAG_PREFIX}_performance_summary.md"

# Optional comma-separated subset for quick reruns. The default covers all
# eight combinations. The cumulative path
# all-off -> stage-out-only -> no-pack -> all-on remains available for ordered
# marginal comparisons, while the other four cells expose interactions.
CELLS=${CELLS:-all-off,stage-out-only,stage-in-only,pack-only,no-stage-out,no-stage-in,no-pack,all-on}

mkdir -p "$LOG_ROOT" "$RESULTS_ROOT"

cell_switches() {
  case "$1" in
    all-off) printf '%s' "off off off" ;;
    stage-out-only) printf '%s' "on off off" ;;
    stage-in-only) printf '%s' "off on off" ;;
    pack-only) printf '%s' "off off on" ;;
    no-stage-out) printf '%s' "off on on" ;;
    no-stage-in) printf '%s' "on off on" ;;
    no-pack) printf '%s' "on on off" ;;
    all-on) printf '%s' "on on on" ;;
    *)
      echo "unknown ablation cell: $1" >&2
      return 2
      ;;
  esac
}

run_cell() {
  local cell=$1
  local stage_out stage_in pack
  read -r stage_out stage_in pack <<<"$(cell_switches "$cell")"
  local tag="${TAG_PREFIX}_${cell}"
  local gpu_log="$LOG_ROOT/${tag}_gpu.csv"
  local monitor_pid=""

  cleanup_monitor() {
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_monitor RETURN

  (
    echo "timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_pct"
    while true; do
      timestamp=$(LC_ALL=C date --iso-8601=ns)
      nvidia-smi \
        --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        while IFS= read -r row; do
          echo "$timestamp,$row"
        done
      sleep 1
    done
  ) >"$gpu_log" 2>&1 &
  monitor_pid=$!

  echo "[$(date --iso-8601=seconds)] START cell=$cell tag=$tag stage_out=$stage_out stage_in=$stage_in pack=$pack" |
    tee -a "$SUMMARY_LOG"
  env \
    -u QW3_KVMEM_PREFILL_WRITEBACK \
    -u QW3_KVMEM_ASSEMBLY_MODE \
    -u QW3_KVMEM_RAW_K_BLOCK_MAJOR \
    -u QW3_KVMEM_RAW_K_TRANSFER_BLOCKS \
    -u QW3_KVMEM_OVERLAP_STAGEIN_ASSEMBLY \
    -u QW3_KVMEM_PERSISTENT_CPU_POOL \
    -u QW3_KVMEM_INCLUSIVE_CPU \
    -u QW3_KVMEM_QUERY_PREFETCH \
    -u QW3_KVMEM_MTP_INCREMENTAL_ASSEMBLY \
    -u QW3_KVMEM_PREFILL_CPU_ADMIT \
    QW3_KVMEM_PERF_TRACE=1 \
    PORT="$PORT" \
    TAG="$tag" \
    KV_DTYPE="$KV_DTYPE" \
    PREFILL_CHUNK="$PREFILL_CHUNK" \
    KVMEM_OPT_STAGE_OUT="$stage_out" \
    KVMEM_OPT_STAGE_IN="$stage_in" \
    KVMEM_OPT_PACK="$pack" \
    KVMEM_BUDGET="$KVMEM_BUDGET" \
    GEN_BUDGET="$GEN_BUDGET" \
    TEMP="$TEMP" \
    THINKING_BUDGET="$THINKING_BUDGET" \
    CPU_GB="$CPU_GB" \
    NVME_GB=0 \
    QUESTION_IDS="$QUESTION_ID" \
    EXPECTED=1 \
    METHOD="kvmem_core_ablation_${cell}_${KV_DTYPE}" \
    "$RUNNER" 2>&1 |
    tee -a "$SUMMARY_LOG"
  echo "[$(date --iso-8601=seconds)] DONE cell=$cell tag=$tag" |
    tee -a "$SUMMARY_LOG"

  cleanup_monitor
  trap - RETURN
}

IFS=',' read -r -a cells <<<"$CELLS"
for cell in "${cells[@]}"; do
  run_cell "$cell"
done

summary_args=()
for cell in "${cells[@]}"; do
  summary_args+=(--cell "$cell=${TAG_PREFIX}_${cell}")
done
"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/summarize_kvmem_perf_ablation.py" \
  --log-root "$LOG_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --output-json "$SUMMARY_JSON" \
  --output-markdown "$SUMMARY_MD" \
  "${summary_args[@]}"

if [[ "$TEMP" == "0" && " ${cells[*]} " == *" all-off "* ]]; then
  "$ROOT/.venv/bin/python" - "$RESULTS_ROOT" "$TAG_PREFIX" "${cells[@]}" <<'PY'
import json
import pathlib
import sys

results_root = pathlib.Path(sys.argv[1])
tag_prefix = sys.argv[2]
cells = sys.argv[3:]

def load(cell):
    path = results_root / f"{tag_prefix}_{cell}" / "answers.jsonl"
    with path.open(encoding="utf-8") as f:
        return json.loads(f.readline())

reference = load("all-off")
fields = ("reasoning", "hypothesis", "raw_response", "finish_reason")
for cell in cells:
    if cell == "all-off":
        continue
    candidate = load(cell)
    different = [
        field for field in fields
        if reference.get(field) != candidate.get(field)
    ]
    if (reference.get("usage", {}).get("completion_tokens") !=
            candidate.get("usage", {}).get("completion_tokens")):
        different.append("usage.completion_tokens")
    if different:
        raise SystemExit(
            f"deterministic KVMem ablation mismatch for {cell}: "
            + ", ".join(different)
        )
print("Deterministic KVMem ablation passed: all cells are byte-identical")
PY
fi

echo "[$(date --iso-8601=seconds)] ABLATION COMPLETE tag_prefix=$TAG_PREFIX" |
  tee -a "$SUMMARY_LOG"
echo "summary_json=$SUMMARY_JSON"
echo "summary_markdown=$SUMMARY_MD"
