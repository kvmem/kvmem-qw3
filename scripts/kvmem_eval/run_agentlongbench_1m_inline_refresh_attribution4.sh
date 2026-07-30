#!/usr/bin/env bash
# Three-arm attribution on four AgentLongBench-1M cases whose answer and
# verification evidence were fully selected but whose cached-KV answer failed.
#
# Arms:
#   cached       historical normal KV + historical recurrent state
#   kv_only      fresh selected normal KV + historical recurrent state
#   kv_and_state fresh selected normal KV + fresh recurrent state
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
TAG=${TAG:-agentlongbench_1m_inline_refresh_attribution4_${STAMP}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
IDS_FILE=${IDS_FILE:-$ROOT/scripts/kvmem_eval/agentlongbench_1m_message_mass_attribution4_ids.txt}
DATASET=${DATASET:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl}
PORT=${PORT:-8089}

mapfile -t sample_ids < <(
  sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "$IDS_FILE"
)
if [[ ${#sample_ids[@]} -ne 4 ]]; then
  echo "expected four sample IDs in $IDS_FILE, got ${#sample_ids[@]}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
if curl -fsS --noproxy '*' \
    "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; refusing to disturb it" >&2
  exit 3
fi

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup_server EXIT INT TERM

question_args=()
for sample_id in "${sample_ids[@]}"; do
  question_args+=(--question-id "$sample_id")
done

for arm in cached kv_only kv_and_state; do
  arm_root="$OUTPUT_ROOT/$arm"
  server_log="$LOG_ROOT/${TAG}_${arm}_server.log"
  run_log="$LOG_ROOT/${TAG}_${arm}_runner.log"
  mkdir -p "$arm_root"
  inline_args=()
  if [[ "$arm" != "cached" ]]; then
    inline_args+=(--kvmem-inline-refresh "$arm")
  fi

  {
    echo "[attribution-config] arm=$arm"
    echo "[attribution-config] temperature=0"
    echo "[attribution-config] selected_context=message_mass_b32_alpha0.5"
    echo "[attribution-config] ids=${sample_ids[*]}"
  } | tee -a "$run_log"

  env \
    -u QW3_KVMEM_REBUILT_STATE_DIR \
    QW3_KVMEM_ENABLE_INLINE_REFRESH=1 \
    QW3_KVMEM_RECOMPUTE_QUERY=1 \
    QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
    QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
    QW3_KVMEM_TRACE=1 \
    QW3_KVMEM_TIMING=1 \
    QW3_KVMEM_DUMP_SCORES="$arm_root/retrieval_scores.jsonl" \
    QW3_Q8_BF16_MAIN=0 \
    QW3_FATTN_NSPLIT=1 \
    QW3_PREFILL_FA2_NSPLIT=1 \
    QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
    "$ROOT/build/qw3" serve \
      --model "$MODEL" \
      --ctx 1310720 \
      --kv-dtype fp8 \
      --kvmem \
      --kvmem-block-tokens 32 \
      --kvmem-budget 229376 \
      --kvmem-gen-budget 32768 \
      --kvmem-sink-blocks 8 \
      --kvmem-recent-blocks 0 \
      --kvmem-method retrieval \
      --kvmem-retrieval-method sub-block-mean-k \
      --kvmem-subblocks 1 \
      --kvmem-subblock-reduce max \
      --kvmem-semantic-expansion message \
      --kvmem-group-score-reduce length-normalized-mass \
      --kvmem-group-length-alpha 0.5 \
      --kvmem-update-mode step \
      --kvmem-query-conditioned \
      --kvmem-immutable-k \
      --kvmem-gpu-memory-ratio 0.5 \
      --kvmem-cpu-gb 80 \
      --enable-thinking \
      --thinking-budget 8192 \
      --prefill-chunk 2048 \
      --temp 0 \
      --native-mtp-speculate \
      --mtp-chain 4 \
      --host 127.0.0.1 \
      --port "$PORT" \
      >"$server_log" 2>&1 &
  server_pid=$!

  healthy=0
  for _ in $(seq 1 600); do
    if curl -fsS --noproxy '*' \
        "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      healthy=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -100 "$server_log" >&2 || true
      exit 4
    fi
    sleep 1
  done
  if [[ "$healthy" -ne 1 ]]; then
    echo "server health timeout for $arm; see $server_log" >&2
    exit 4
  fi

  for expected in \
    'kvmem_semantic_expansion=message' \
    'kvmem_group_score_reduce=length-normalized-mass' \
    'kvmem_group_length_alpha=0.5' \
    'kvmem_recompute_query=1' \
    'kvmem_immutable_k=1'; do
    if ! grep -q "$expected" "$server_log"; then
      echo "server did not confirm required setting: $expected" >&2
      exit 5
    fi
  done

  "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
      --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
      --dataset "$DATASET" \
      --manifest "$MANIFEST" \
      --allow-custom-subset \
      --benchmark-name AgentLongBench-1M-inline-refresh-attribution4 \
      --output-root "$arm_root" \
      --api-base "http://127.0.0.1:$PORT/v1" \
      --model "$(basename "$MODEL")" \
      --method "kvmem_message_b32_mass_alpha05_${arm}_temp0" \
      --temperature 0 \
      --top-p 0.95 \
      --max-tokens 32768 \
      --context-window 1310720 \
      --context-safety-margin 16 \
      --timeout-sec 7200 \
      --max-sample-sec 7200 \
      --attempts 1 \
      --enable-thinking \
      --seed 20260722 \
      --kvmem-retrieval-trace-metadata \
      --kvmem-message-expansion \
      "${inline_args[@]}" \
      "${question_args[@]}" \
      2>&1 | tee -a "$run_log"

  "$ROOT/.venv/bin/python" - "$arm_root/validation_report.json" "$arm" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    report.get("passed") is not True
    or report.get("answers_unique") != 4
    or report.get("eval_unique") != 4
):
    raise SystemExit(f"{sys.argv[2]} validation failed: {report}")
print(f"{sys.argv[2]} validation passed: 4 answers and 4 evaluations")
PY
  cleanup_server
done

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/summarize_agentlongbench_inline_refresh_attribution.py" \
    --root "$OUTPUT_ROOT" \
    --ids-file "$IDS_FILE"

echo "[complete] output=$OUTPUT_ROOT"
