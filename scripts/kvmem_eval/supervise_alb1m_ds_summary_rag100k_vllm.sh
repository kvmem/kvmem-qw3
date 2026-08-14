#!/usr/bin/env bash
set -euo pipefail

SUMMARY_FILE="${SUMMARY_FILE:-/data/chaidi/kvmem_eval/data/deepseek_million_summary/summary_plus_tail.jsonl}"
DATASET="${DATASET:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}"
OLD_ROOT="${OLD_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_compact_rag_notail_cap100k_vllm_fp8_mtp4_20260811}"
OUT_ROOT="${OUT_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds_summary_rag_notail_cap100k_vllm_fp8_mtp4_20260811}"
LOG="${LOG:-/data/chaidi/kvmem_eval/logs/agentlongbench_1m_ds_summary_rag_notail_cap100k_vllm_fp8_mtp4_20260811.log}"
API_BASE="${API_BASE:-http://127.0.0.1:18121/v1}"
MODEL="${MODEL:-Qwen3.6-27B-FP8}"
PYTHON="${PYTHON:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}"
RUNNER="${RUNNER:-/home/chaidi/AgentLongBench-Long/script/deepseekMillion/run_no_tail_cap100k_1m.py}"
TOKENIZER="${TOKENIZER:-/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/chaidi/kvmem_eval/models/jinaai/jina-embeddings-v2-small-en}"
WORKERS="${WORKERS:-4}"

mkdir -p "$(dirname "$LOG")" "$OUT_ROOT"
while [[ ! -r "$SUMMARY_FILE" ]]; do
  printf '[waiting-summary] %s\n' "$SUMMARY_FILE" | tee -a "$LOG"
  sleep 30
done

# The rankings are independent of the compact summary.  Reuse the already
# materialized full-history B32/O8 Jina rankings, but keep all answer and
# checkpoint artifacts in a new experiment root.
if [[ ! -d "$OUT_ROOT/retrieval" ]]; then
  mkdir -p "$OUT_ROOT/retrieval"
  cp -al "$OLD_ROOT/retrieval/." "$OUT_ROOT/retrieval/"
fi

if ! curl -fsS "$API_BASE/models" >/dev/null; then
  printf '[error] vLLM service is not available at %s\n' "$API_BASE" | tee -a "$LOG"
  exit 1
fi

cd /home/chaidi/AgentLongBench-Long
set -o pipefail
env \
  PYTHONNOUSERSITE=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  "$PYTHON" "$RUNNER" \
    --dataset "$DATASET" \
    --source-compact-root "$SUMMARY_FILE" \
    --output-root "$OUT_ROOT" \
    --api-base "$API_BASE" \
    --api-backend vllm \
    --model "$MODEL" \
    --tokenizer-path "$TOKENIZER" \
    --embedding-model-path "$EMBEDDING_MODEL" \
    --rag-chunk-tokenizer-path "$TOKENIZER" \
    --embedding-device cuda \
    --embedding-batch-size 256 \
    --workers "$WORKERS" \
    --limit 50 \
  2>&1 | tee -a "$LOG"
