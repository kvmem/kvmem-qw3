#!/usr/bin/env bash
# Continue automatically after the already-running capture service: validate the
# exact 19 dumps, calculate gold coverage, run dense selected-text replay, and
# join all outcomes into one diagnostic report.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CAPTURE_UNIT=${CAPTURE_UNIT:-qw3-agentlongbench-1m-error19-capture-20260727.service}
CAPTURE_TAG=${CAPTURE_TAG:-agentlongbench_1m_ds50_kvmem_error19_selected_dump_20260727_103114}
CAPTURE_ROOT=${CAPTURE_ROOT:-/data/chaidi/kvmem_eval/results/$CAPTURE_TAG}
REPLAY_TAG=${REPLAY_TAG:-${CAPTURE_TAG}_selected_text_dense_fp8}
REPLAY_ROOT=${REPLAY_ROOT:-/data/chaidi/kvmem_eval/results/$REPLAY_TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
PIPELINE_LOG=${PIPELINE_LOG:-$LOG_ROOT/${CAPTURE_TAG}_pipeline.log}
DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl
KVMEM_ROOT=/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds50_k224k_g32k_b32_meank_kvfp8_idxqfp16_canonical_refresh1_t06_think8k_cpu_all_on_gpuratio050_20260726_235009
BASE=/home/chaidi/AgentLongBench-Long/results

mkdir -p "$LOG_ROOT"
while systemctl --user is-active --quiet "$CAPTURE_UNIT"; do
  sleep 30
done

"$ROOT/.venv/bin/python" - "$CAPTURE_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
dump = root / "kvmem_retrieval_dump.jsonl"
manifest = root / "capture_manifest.jsonl"
tags = [
    json.loads(line)["trace_tag"]
    for line in dump.open(encoding="utf-8")
    if line.strip() and json.loads(line).get("type") == "meta"
]
manifest_ids = [
    json.loads(line)["stable_sample_id"]
    for line in manifest.open(encoding="utf-8")
    if line.strip()
]
if len(tags) != 19 or len(set(tags)) != 19 or tags != manifest_ids:
    raise SystemExit(
        f"capture validation failed: meta={len(tags)} "
        f"unique={len(set(tags))} manifest={len(manifest_ids)}"
    )
print("capture validation passed: 19 exact ordered dumps")
PY

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/analyze_agentlongbench_gold_block_coverage.py" \
  --dataset "$DATA" \
  --kvmem-dump "$CAPTURE_ROOT/kvmem_retrieval_dump.jsonl" \
  --output-root "$CAPTURE_ROOT/gold_coverage" \
  2>&1 | tee -a "$PIPELINE_LOG"

CAPTURE_TAG="$CAPTURE_TAG" TAG="$REPLAY_TAG" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_error19_selected_text.sh" \
  2>&1 | tee -a "$PIPELINE_LOG"

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
  2>&1 | tee -a "$PIPELINE_LOG"
