#!/usr/bin/env bash
set -euo pipefail

# Matched to the canonical 500-sample Mean-K baseline
# s500_32k_bt32_meank_unified_eval_20260712_143132. The only intended
# retrieval change is Mean-K -> Adaptive Multi-Prototype Mean-K, using the
# standard 0.10/0.06 marginal-gain thresholds.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
TAG="${TAG:-s500_32k_bt32_keydir_adaptive_g010_006}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY must be set for the deepseek-v4-pro judge" >&2
  exit 2
fi

exec "$PYTHON" "$ROOT/scripts/kvmem_eval/run_kvmem_eval.py" \
  --tag "$TAG" \
  --binary "$ROOT/build/qw3" \
  --model "$ROOT/models/Qwen3.6-27B-Q8_0.gguf" \
  --data /data/chaidi/kvmem_eval/data/longmemeval_s.json \
  --out-dir /data/chaidi/kvmem_eval/results \
  --host 127.0.0.1 \
  --port 8086 \
  --ctx 262144 \
  --kv-dtype fp16 \
  --prefill-chunk 2048 \
  --block-tokens 32 \
  --budget 32768 \
  --gen-budget 32768 \
  --sink-blocks 8 \
  --recent-blocks 0 \
  --method retrieval \
  --retrieval-method key-direction-adaptive \
  --adaptive-gain-1to2 0.10 \
  --adaptive-gain-2to4 0.06 \
  --update-mode step \
  --optimization-level default \
  --query-conditioned \
  --gpu-memory-ratio 0.5 \
  --cpu-gb 64 \
  --nvme-gb 256 \
  --thinking \
  --thinking-budget 4096 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-tokens 32768 \
  --mtp \
  --mtp-chain 4 \
  "$@"
