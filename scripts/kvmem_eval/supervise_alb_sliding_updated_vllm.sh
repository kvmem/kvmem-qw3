#!/usr/bin/env bash
set -euo pipefail

# vLLM 0.21 produced a reproducible FlashInfer CUDA illegal-memory-access on
# a 66K-token MTP request.  vLLM 0.22 has exact tokenizer parity and completed
# the earlier long baseline services without that runtime failure.
VLLM_ENV=${VLLM_ENV:-/home/chaidi/qw3/.venv}
VLLM_BIN=${VLLM_BIN:-$VLLM_ENV/bin/vllm}
# Sliding Window does not need the separate Jina/RAG environment.  Render and
# crop prompts with the exact Transformers build used by the serving process;
# otherwise Unicode-dependent tokenizer drift can violate the strict active
# prompt cap even when a small fixed reserve is present.
RUN_PY=${RUN_PY:-$VLLM_ENV/bin/python}
RUNNER=${RUNNER:-/home/chaidi/AgentLongBench-Long/script/slidingWindow/run_sliding_window.py}
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/e89b16ebf1988b3d6befa7de50abc2d76f26eb09}
MODEL_NAME=${MODEL_NAME:-Qwen3.6-27B-FP8}
TOKENIZER=${TOKENIZER:-/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8}
# The similarly named file under AgentLongBench-Long/results is only a Git-LFS
# pointer in this checkout.  Use the materialized canonical 100-row cohort that
# also backs the strict Compact+RAG and KVMem utility runs.
DATA_512=${DATA_512:-/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl}
DATA_1M=${DATA_1M:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
OUT_512=${OUT_512:-/data/chaidi/kvmem_eval/results/agentlongbench_512k_sliding_window_cap64k_vllm022_fp8_mtp4_20260812}
OUT_1M=${OUT_1M:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_sliding_window_cap100k_vllm022_fp8_mtp4_20260812}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
PORT=${PORT:-18123}
API_BASE="http://127.0.0.1:${PORT}/v1"

mkdir -p "$OUT_512" "$OUT_1M" "$LOG_ROOT"
server_log="$LOG_ROOT/agentlongbench_sliding_updated_vllm022_fp8_mtp4_server_20260812.log"

for output_root in "$OUT_512" "$OUT_1M"; do
  env OUTPUT_ROOT="$output_root" MODEL_PATH="$MODEL_PATH" \
    MODEL_NAME="$MODEL_NAME" TOKENIZER="$TOKENIZER" \
    "$VLLM_ENV/bin/python" - <<'PY'
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch
import transformers
import vllm

gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
     "--format=csv,noheader,nounits"], text=True,
).splitlines()[0]
payload = {
    "schema_version": 1,
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": platform.node(),
    "gpu": gpu,
    "python": sys.executable,
    "vllm_version": vllm.__version__,
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__,
    "model_path": str(Path(os.environ["MODEL_PATH"]).resolve()),
    "model_name": os.environ["MODEL_NAME"],
    "tokenizer_path": str(Path(os.environ["TOKENIZER"]).resolve()),
}
path = Path(os.environ["OUTPUT_ROOT"]) / "summary" / "serving_environment.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
done

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT

env \
  PATH="$VLLM_ENV/bin:$PATH" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tokenizer "$TOKENIZER" \
    --host 127.0.0.1 --port "$PORT" \
    --language-model-only \
    --safetensors-load-strategy prefetch \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --async-scheduling \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    --compilation-config '{"mode":3}' \
    >"$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 240); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -120 "$server_log" >&2
    exit 1
  fi
  if curl --noproxy '*' -fsS "$API_BASE/models" >/dev/null; then
    break
  fi
  sleep 2
done
curl --noproxy '*' -fsS "$API_BASE/models" >/dev/null

COMMON=(
  --api-base "$API_BASE"
  --model "$MODEL_NAME"
  --api-backend vllm
  --tokenizer-path "$TOKENIZER"
  --temperature 0.6
  --top-p 0.95
  --top-k 20
  --thinking-budget 8192
  --answer-max-tokens 32768
  --chat-template-reserve 16
  --seed 20260722
  --continue-on-error
  --timeout-sec 14400
  --max-call-sec 14400
  --attempts 2
)

set +e
env PYTHONUNBUFFERED=1 "$RUN_PY" "$RUNNER" run \
  --dataset-path "$DATA_512" \
  --output-root "$OUT_512" \
  --window-prompt-tokens 65536 \
  "${COMMON[@]}" \
  >"$LOG_ROOT/agentlongbench_512k_sliding_cap64k_20260812.log" 2>&1 &
pid_512=$!

env PYTHONUNBUFFERED=1 "$RUN_PY" "$RUNNER" run \
  --dataset-path "$DATA_1M" \
  --output-root "$OUT_1M" \
  --window-prompt-tokens 102400 \
  --limit 50 \
  "${COMMON[@]}" \
  >"$LOG_ROOT/agentlongbench_1m_sliding_cap100k_20260812.log" 2>&1 &
pid_1m=$!

wait "$pid_512"; status_512=$?
wait "$pid_1m"; status_1m=$?
set -e

stop_server

for spec in "$DATA_512|$OUT_512|65536|" "$DATA_1M|$OUT_1M|102400|50"; do
  IFS='|' read -r data out cap limit <<<"$spec"
  limit_args=()
  if [[ -n "$limit" ]]; then
    limit_args=(--limit "$limit")
  fi
  "$RUN_PY" "$RUNNER" summarize \
    --dataset-path "$data" --output-root "$out" \
    --window-prompt-tokens "$cap" \
    --api-backend vllm --tokenizer-path "$TOKENIZER" \
    --temperature 0.6 --top-p 0.95 --top-k 20 \
    --thinking-budget 8192 --answer-max-tokens 32768 \
    --chat-template-reserve 16 --seed 20260722 \
    "${limit_args[@]}" \
    >"$out/summary/summarize.stdout.json"
done

if [[ "$status_512" -ne 0 || "$status_1m" -ne 0 ]]; then
  exit 2
fi
