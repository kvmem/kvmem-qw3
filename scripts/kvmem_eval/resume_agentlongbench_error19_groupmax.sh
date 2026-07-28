#!/usr/bin/env bash
# Resume the interleaved error-19 pipeline against already resident servers.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CAPTURE_ROOT=/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds50_kvmem_error19_selected_dump_20260727_103114
REPLAY_ROOT=${CAPTURE_ROOT}_selected_text_dense_fp8_interleaved
KVMEM_EVAL=/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds50_k224k_g32k_b32_meank_kvfp8_idxqfp16_canonical_refresh1_t06_think8k_cpu_all_on_gpuratio050_20260726_235009/eval.jsonl
LOG=/data/chaidi/kvmem_eval/logs/agentlongbench_1m_error19_groupmax_resume_20260727.log

exec "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_error19_interleaved.py" \
  --ids-file "$ROOT/scripts/kvmem_eval/agentlongbench_1m_kvmem_error19_ids.txt" \
  --dataset /home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
  --dump-file "$CAPTURE_ROOT/kvmem_retrieval_dump.jsonl" \
  --capture-root "$CAPTURE_ROOT" \
  --replay-root "$REPLAY_ROOT" \
  --kvmem-eval "$KVMEM_EVAL" \
  --kvmem-api http://127.0.0.1:8088/v1 \
  --dense-api http://127.0.0.1:8087/v1 \
  --model Qwen3.6-27B-Q8_0.gguf \
  --timeout-sec 7200 \
  --group-max-first 3 \
  >>"$LOG" 2>&1
