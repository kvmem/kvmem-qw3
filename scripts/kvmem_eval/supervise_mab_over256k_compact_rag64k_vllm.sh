#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
VLLM_ENV=${VLLM_ENV:-/home/chaidi/qw3/.venv}
VLLM_BIN=${VLLM_BIN:-$VLLM_ENV/bin/vllm}
RUN_PY=${RUN_PY:-$VLLM_ENV/bin/python}
RUNNER=${RUNNER:-$ROOT/scripts/kvmem_eval/run_memoryagentbench_baselines.py}
SCORER=${SCORER:-$ROOT/scripts/kvmem_eval/score_memoryagentbench_official_local.py}
JUDGE=${JUDGE:-$ROOT/scripts/kvmem_eval/judge_memoryagentbench_special.py}
WORKSPACE=${WORKSPACE:-/data/chaidi/kvmem_eval/results/memoryagentbench_over256k_compact_rag_notail_cap64k_vllm021_20260812}
SOURCE_WORKSPACE=${SOURCE_WORKSPACE:-/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802}
REFERENCE_RESULTS=${REFERENCE_RESULTS:-/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full}
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/e89b16ebf1988b3d6befa7de50abc2d76f26eb09}
MODEL_NAME=${MODEL_NAME:-Qwen3.6-27B-FP8}
TOKENIZER=${TOKENIZER:-/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8}
RAG_MODEL=${RAG_MODEL:-/data/chaidi/kvmem_eval/models/jinaai/jina-embeddings-v2-small-en}
PORT=${PORT:-18122}
API_BASE="http://127.0.0.1:${PORT}/v1"

mkdir -p "$WORKSPACE/logs" "$WORKSPACE/run" "$WORKSPACE/shared/rag"

COMMON=(
  --workspace "$WORKSPACE"
  --reference-results "$REFERENCE_RESULTS"
  --min-context-tokens-exclusive 262144
  --context-window 262144
  --generation-reserve 32768
  --strict-final-prompt-tokens 65536
  --sliding-prompt-tokens 65536
  --compact-full-history
  --compact-no-tail
  --source-shared-workspace "$SOURCE_WORKSPACE"
  --rag-dynamic-budget
  --rag-no-metadata
  --rag-top-k 2200
  --rag-block-size 32
  --rag-overlap 8
  --rag-model "$RAG_MODEL"
  --tokenizer-dir "$TOKENIZER"
  --embedding-batch-size 256
)

prepared_rag=$(find "$WORKSPACE/shared/rag" -maxdepth 1 -type f 2>/dev/null | wc -l)
if [[ "$prepared_rag" -lt 30 ]]; then
  env \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    "$PY" "$RUNNER" prepare-rag \
      "${COMMON[@]}" \
      --embedding-device cuda \
      >"$WORKSPACE/logs/prepare_rag.log" 2>&1
fi

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT

vllm_version=$(
  "$VLLM_ENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null \
    | tr -cd '[:alnum:]._-'
)
server_log="$WORKSPACE/logs/server_vllm${vllm_version:-unknown}_fp8_mtp4.log"
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
echo "$server_pid" >"$WORKSPACE/run/server.pid"

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

set +e
env PYTHONUNBUFFERED=1 "$RUN_PY" "$RUNNER" run \
  "${COMMON[@]}" \
  --api-base "$API_BASE" \
  --model-name "$MODEL_NAME" \
  --server-mtp-chain 4 \
  --prefix-cache-guard-pages 0 \
  --method compact-rag \
  >"$WORKSPACE/logs/compact_rag.log" 2>&1
run_status=$?
if [[ "$run_status" -ne 0 ]]; then
  set -e
  stop_server
  exit 2
fi
env PYTHONUNBUFFERED=1 "$RUN_PY" "$RUNNER" run \
  "${COMMON[@]}" \
  --api-base "$API_BASE" \
  --model-name "$MODEL_NAME" \
  --server-mtp-chain 4 \
  --prefix-cache-guard-pages 0 \
  --method sliding-window \
  >"$WORKSPACE/logs/sliding_window.log" 2>&1
sliding_status=$?
set -e
stop_server

for method in compact-rag sliding-window; do
  "$PY" "$SCORER" \
    --results-dir "$WORKSPACE/methods/$method" \
    --allow-partial \
    >"$WORKSPACE/logs/score_${method}.log" 2>&1

  if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      "$PY" "$JUDGE" \
        --results-dir "$WORKSPACE/methods/$method" \
        --allow-partial \
        >"$WORKSPACE/logs/judge_${method}.log" 2>&1
  else
    printf 'DEEPSEEK_API_KEY is unset; special judge deferred.\n' \
      >"$WORKSPACE/logs/judge_${method}.pending"
  fi
done

if [[ "$sliding_status" -ne 0 ]]; then
  exit 2
fi
