#!/usr/bin/env bash
# Two-resident-server, one-sample-at-a-time error-19 diagnostic pipeline.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CAPTURE_TAG=${CAPTURE_TAG:-agentlongbench_1m_ds50_kvmem_error19_selected_dump_20260727_103114}
REPLAY_TAG=${REPLAY_TAG:-${CAPTURE_TAG}_selected_text_dense_fp8_interleaved}
CAPTURE_ROOT=${CAPTURE_ROOT:-/data/chaidi/kvmem_eval/results/$CAPTURE_TAG}
REPLAY_ROOT=${REPLAY_ROOT:-/data/chaidi/kvmem_eval/results/$REPLAY_TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
KVMEM_LOG=${KVMEM_LOG:-$LOG_ROOT/${REPLAY_TAG}_kvmem_server.log}
DENSE_LOG=${DENSE_LOG:-$LOG_ROOT/${REPLAY_TAG}_dense_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${REPLAY_TAG}_runner.log}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
IDS_FILE=$ROOT/scripts/kvmem_eval/agentlongbench_1m_kvmem_error19_ids.txt
DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl
KVMEM_EVAL=/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds50_k224k_g32k_b32_meank_kvfp8_idxqfp16_canonical_refresh1_t06_think8k_cpu_all_on_gpuratio050_20260726_235009/eval.jsonl

mkdir -p "$CAPTURE_ROOT" "$REPLAY_ROOT" "$LOG_ROOT"
for port in 8087 8088; do
  if curl -fsS --noproxy '*' "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "port $port already has a healthy server; refusing to disturb it" >&2
    exit 3
  fi
done

kvmem_pid=""
dense_pid=""
cleanup() {
  for pid in "$dense_pid" "$kvmem_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

env \
  -u QW3_KVMEM_REBUILT_STATE_DIR \
  QW3_KVMEM_RECOMPUTE_QUERY=1 \
  QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_KVMEM_DUMP_SCORES="$CAPTURE_ROOT/kvmem_retrieval_dump.jsonl" \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 1310720 --kv-dtype fp8 \
    --kvmem --kvmem-block-tokens 32 \
    --kvmem-budget 229376 --kvmem-gen-budget 32768 \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-immutable-k --kvmem-gpu-memory-ratio 0.5 \
    --kvmem-cpu-gb 80 \
    --enable-thinking --thinking-budget 8192 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port 8088 \
    >"$KVMEM_LOG" 2>&1 &
kvmem_pid=$!

for _ in $(seq 1 300); do
  if curl -fsS --noproxy '*' "http://127.0.0.1:8088/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$kvmem_pid" 2>/dev/null; then
    tail -100 "$KVMEM_LOG" >&2 || true
    exit 4
  fi
  sleep 2
done
curl -fsS --noproxy '*' "http://127.0.0.1:8088/health" >/dev/null

env \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 262144 --kv-dtype fp8 \
    --enable-thinking --thinking-budget 8192 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port 8087 \
    >"$DENSE_LOG" 2>&1 &
dense_pid=$!

for _ in $(seq 1 300); do
  if curl -fsS --noproxy '*' "http://127.0.0.1:8087/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$dense_pid" 2>/dev/null; then
    tail -100 "$DENSE_LOG" >&2 || true
    exit 5
  fi
  sleep 2
done
curl -fsS --noproxy '*' "http://127.0.0.1:8087/health" >/dev/null

echo "[dual-server-ready]" | tee -a "$RUN_LOG"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader |
  tee -a "$RUN_LOG"

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_error19_interleaved.py" \
  --ids-file "$IDS_FILE" \
  --dataset "$DATA" \
  --dump-file "$CAPTURE_ROOT/kvmem_retrieval_dump.jsonl" \
  --capture-root "$CAPTURE_ROOT" \
  --replay-root "$REPLAY_ROOT" \
  --kvmem-eval "$KVMEM_EVAL" \
  --kvmem-api http://127.0.0.1:8088/v1 \
  --dense-api http://127.0.0.1:8087/v1 \
  --model "$(basename "$MODEL")" \
  --timeout-sec 7200 \
  2>&1 | tee -a "$RUN_LOG"

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/analyze_agentlongbench_gold_block_coverage.py" \
  --dataset "$DATA" \
  --kvmem-dump "$CAPTURE_ROOT/kvmem_retrieval_dump.jsonl" \
  --output-root "$CAPTURE_ROOT/gold_coverage" \
  2>&1 | tee -a "$RUN_LOG"

BASE=/home/chaidi/AgentLongBench-Long/results
KVMEM_ROOT=/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds50_k224k_g32k_b32_meank_kvfp8_idxqfp16_canonical_refresh1_t06_think8k_cpu_all_on_gpuratio050_20260726_235009
"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/summarize_agentlongbench_error19_diagnostic.py" \
  --coverage "$CAPTURE_ROOT/gold_coverage/per_sample_gold_coverage.jsonl" \
  --selected-text-eval "$REPLAY_ROOT/eval.jsonl" \
  --selected-text-answers "$REPLAY_ROOT/answers.jsonl" \
  --kvmem-eval "$KVMEM_ROOT/eval.jsonl" \
  --kvmem-answers "$KVMEM_ROOT/answers.jsonl" \
  --compact-only-eval "$BASE/deepseek_million_qwen_q8_compact_only_50/eval/compact_only.eval.jsonl" \
  --rag1024-eval "$BASE/deepseek_million_qwen_q8_compact_rag_jina_t30_b1024_o128_50/eval/compact_rag.eval.jsonl" \
  --rag32-eval "$BASE/deepseek_million_qwen_q8_compact_rag_jina_t960_b32_o8_50/eval/compact_rag.eval.jsonl" \
  --sliding-eval "$BASE/deepseek_million_qwen_q8_sliding_window_32k_50/eval/sliding_window.eval.jsonl" \
  --output-root "$CAPTURE_ROOT/diagnostic" \
  2>&1 | tee -a "$RUN_LOG"
