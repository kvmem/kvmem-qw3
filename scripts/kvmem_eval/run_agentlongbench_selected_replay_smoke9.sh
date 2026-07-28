#!/usr/bin/env bash
# Dense text-replay control. By default it runs the nine KVMem-correct direct
# retrieval-gap cases; set ALL_DUMP=1 to run all snapshots in the frozen dump.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8087}
TAG=${TAG:-agentlongbench_kvmem32k_selected_text_replay_direct9_20260718}
MODEL=${MODEL:-${ROOT}/models/Qwen3.6-27B-Q8_0.gguf}
RESULT=${RESULT:-/data/chaidi/kvmem_eval/results/${TAG}}
LOG_DIR=${LOG_DIR:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${LOG_DIR}/${TAG}_server.log
RUN_LOG=${LOG_DIR}/${TAG}_runner.log

mkdir -p "$RESULT" "$LOG_DIR"

if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "port ${PORT} already has a healthy server; refusing to reuse it" >&2
  exit 1
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
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 65536 --kv-dtype fp16 \
    --enable-thinking --thinking-budget 4096 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" >>"$SERVER_LOG" 2>&1 &
server_pid=$!

healthy=0
for _ in $(seq 1 180); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null; then
    healthy=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$healthy" -ne 1 ]]; then
  echo "dense qw3 health timeout" >&2
  tail -100 "$SERVER_LOG" >&2
  exit 1
fi

ids=(
  ec997131fbdc5b28e192ff49d27fc7c27a299c236eed47424ebb4373331a9a2a
  b577e3b1bf2c20e70dba3873a541785fa65ee53efbcb6ff9fb2420467dc9113c
  a792236330365fb213c1d89e733836d42df8183ffee485bb9993ba78b4dbe3ac
  a3011fb9099e05f563637a02de6f25b3c8128a6d61422ccb022a380a3bd0ce0f
  3f44a5bfbcd6a6b0dae8f5227ed3fb4e7f3814d34fad94826da126d291b4432b
  244e85f424ae0dd83cd64385515ca2a70f0b92fd2dc48c804dcd6261c14576fa
  9f6b35f6f282170de3c82415d74c914432422b6c18e7963db19a71e9dca72779
  46728d7a51ffd7ed4b9edb588d270dbefc7479e6aed3c5462b7f674ab422b9f6
  bc3fec1b8a54460dd3969e94e52e4b0b2c966d425164d1df07491a73224e382f
)
question_args=()
if [[ "${ALL_DUMP:-0}" != "1" ]]; then
  for sid in "${ids[@]}"; do
    question_args+=(--question-id "$sid")
  done
fi

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_selected_replay.py" \
  --output-root "$RESULT" \
  --api-base "http://127.0.0.1:${PORT}/v1" \
  --model "$(basename "$MODEL")" \
  --temperature 0.6 --top-p 0.95 --max-tokens 32768 \
  --thinking-budget 4096 --context-window 65536 \
  "${question_args[@]}" 2>&1 | tee -a "$RUN_LOG"
