#!/usr/bin/env bash
# Full AgentLongBench-1M canonical50 evaluation for the message-expansion /
# length-normalized-mass retrieval method. The four completed controlled-probe
# rows are imported into the destination, so only the remaining 46 samples are
# generated while the final summary still covers the canonical 50.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
TAG=${TAG:-agentlongbench_1m_message_mass_b32_full50_${STAMP}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_runner.log}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
DATASET=${DATASET:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl}
CANONICAL50_MANIFEST=${CANONICAL50_MANIFEST:-/home/chaidi/AgentLongBench-Long/results/deepseek_million_qwen_q8_compact_only_50/manifests/compact_only.manifest.jsonl}
SEED_ROOT=${SEED_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_message_mass_b32_oracle4_20260729_072653}
PORT=${PORT:-8088}
BENCHMARK_NAME=AgentLongBench-1M-message-mass-full50
METHOD=kvmem_message_b32_mass_alpha05_query_replay_immutable_refresh1

mapfile -t sample_ids < <(
  "$ROOT/.venv/bin/python" - "$CANONICAL50_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            print(json.loads(line)["stable_sample_id"])
PY
)
if [[ ${#sample_ids[@]} -ne 50 ]]; then
  echo "expected 50 canonical sample IDs, got ${#sample_ids[@]}" >&2
  exit 2
fi

if [[ ! -e "$OUTPUT_ROOT" ]]; then
  "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/kvmem_eval/seed_agentlongbench_resume.py" \
      --dataset "$DATASET" \
      --selected-id-source "$CANONICAL50_MANIFEST" \
      --output-root "$OUTPUT_ROOT" \
      --source-root "$SEED_ROOT" \
      --benchmark-name "$BENCHMARK_NAME" \
      --method "$METHOD"
elif [[ ! -e "$OUTPUT_ROOT/run_config.json" &&
        ! -e "$OUTPUT_ROOT/resume_seed_provenance.json" ]]; then
  echo "refusing an unrecognized non-empty output root: $OUTPUT_ROOT" >&2
  exit 3
fi

mkdir -p "$LOG_ROOT"
if curl -fsS --noproxy '*' \
    "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; refusing to disturb it" >&2
  exit 3
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

env \
  -u QW3_KVMEM_REBUILT_STATE_DIR \
  QW3_KVMEM_RECOMPUTE_QUERY=1 \
  QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_KVMEM_DUMP_SCORES="$OUTPUT_ROOT/retrieval_scores.jsonl" \
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
    --temp 0.6 \
    --native-mtp-speculate \
    --mtp-chain 4 \
    --host 127.0.0.1 \
    --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

healthy=0
for _ in $(seq 1 600); do
  if curl -fsS --noproxy '*' \
      "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2 || true
    exit 4
  fi
  sleep 1
done
if [[ "$healthy" -ne 1 ]]; then
  echo "server health timeout; see $SERVER_LOG" >&2
  exit 4
fi

for expected in \
  'kvmem_semantic_expansion=message' \
  'kvmem_group_score_reduce=length-normalized-mass' \
  'kvmem_group_length_alpha=0.5' \
  'kvmem_recompute_query=1' \
  'kvmem_immutable_k=1'; do
  if ! grep -q "$expected" "$SERVER_LOG"; then
    echo "server did not confirm required setting: $expected" >&2
    exit 5
  fi
done

question_args=()
for sample_id in "${sample_ids[@]}"; do
  question_args+=(--question-id "$sample_id")
done

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
    --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --allow-custom-subset \
    --benchmark-name "$BENCHMARK_NAME" \
    --output-root "$OUTPUT_ROOT" \
    --api-base "http://127.0.0.1:$PORT/v1" \
    --model "$(basename "$MODEL")" \
    --method "$METHOD" \
    --temperature 0.6 \
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
    "${question_args[@]}" \
    2>&1 | tee "$RUN_LOG"

"$ROOT/.venv/bin/python" - "$OUTPUT_ROOT/validation_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    report.get("passed") is not True
    or report.get("answers_unique") != 50
    or report.get("eval_unique") != 50
):
    raise SystemExit(f"AgentLongBench validation failed: {report}")
print("AgentLongBench validation passed: 50 answers and 50 evaluations")
PY
