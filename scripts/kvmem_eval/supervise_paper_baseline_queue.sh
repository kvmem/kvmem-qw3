#!/usr/bin/env bash
# Complete the controlled baseline queue without dropping terminal length rows.
set -euo pipefail

ALB_PID=${ALB_PID:-3120801}
ROOT=${ROOT:-/home/chaidi/qw3}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
ALB_SCRIPT=${ALB_SCRIPT:-/home/chaidi/AgentLongBench-Long/script/deepseekMillion/run_no_tail_cap100k_1m.py}
ALB_OUTPUT=${ALB_OUTPUT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds_summary_rag_notail_cap100k_vllm022_fp8_mtp4_seq2_20260811}
API_BASE=${API_BASE:-http://127.0.0.1:18121/v1}

while kill -0 "$ALB_PID" 2>/dev/null; do
  sleep 30
done

# The first pass intentionally records failures.  The current runner preserves
# finish_reason=length as a scored terminal prediction, skips every completed
# row, and therefore repairs only missing rows before the model is unloaded.
env PYTHONUNBUFFERED=1 "$PY" "$ALB_SCRIPT" \
  --dataset /home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
  --source-compact-root /home/chaidi/AgentLongBench-Long/summary_plus_tail.jsonl \
  --output-root "$ALB_OUTPUT" \
  --api-base "$API_BASE" \
  --api-backend vllm \
  --model Qwen3.6-27B-FP8 \
  --tokenizer-path /home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8 \
  --embedding-model-path /data/chaidi/kvmem_eval/models/jinaai/jina-embeddings-v2-small-en \
  --rag-chunk-tokenizer-path /home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8 \
  --embedding-device cuda \
  --embedding-batch-size 256 \
  --workers 1 \
  --limit 50

pids=$(pgrep -f '[v]llm serve.*--port 18121' || true)
if [[ -n "$pids" ]]; then
  kill $pids 2>/dev/null || true
  sleep 10
fi

"$ROOT/scripts/kvmem_eval/supervise_mab_over256k_compact_rag64k_vllm.sh"
"$ROOT/scripts/kvmem_eval/supervise_alb_sliding_updated_vllm.sh"
