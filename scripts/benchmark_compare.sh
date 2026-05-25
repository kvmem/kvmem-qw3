#!/usr/bin/env bash
set -euo pipefail

MODEL=""
LLAMA_CLI="llama-completion"
QW3_BIN="./build-cuda/qw3"
PROMPT="Write a concise CUDA kernel optimization checklist."
CTX=32768
THREADS=0
NGL=-1
BATCH=2048
DECODE_TOKENS=32
NATIVE_TOKEN_ID=0
OUT_DIR="${PWD}/bench-out"

usage() {
  cat <<'EOF'
Usage: scripts/benchmark_compare.sh --model MODEL.gguf [options]

Options:
  --model PATH                 GGUF model path (required)
  --llama-cli PATH             llama.cpp binary path (default: llama-completion)
  --qw3-bin PATH               qw3 binary path (default: ./build-cuda/qw3)
  --prompt TEXT                Prompt text for benchmark
  --ctx N                      Context size (default: 32768)
  --threads N                  Threads for llama backend (default: 0)
  --ngl N                      GPU layers for llama backend (default: -1)
  --batch N                    Batch size for llama backend (default: 2048)
  --decode-tokens N            Tokens for steady decode benchmark (default: 32)
  --native-token-id N          Token id for qwen-native loop (default: 0)
  --out-dir PATH               Output directory for logs/results
  -h, --help                   Show help

Notes:
  - qwen-native currently runs a single-token forward per invocation.
  - TTFT for qwen-native is measured as one full invocation latency.
  - qwen-native decode tok/s is approximated by N repeated single-token runs.
EOF
}

need_arg() {
  local name="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    echo "missing value for ${name}" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) need_arg "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
    --llama-cli) need_arg "$1" "${2:-}"; LLAMA_CLI="$2"; shift 2 ;;
    --qw3-bin) need_arg "$1" "${2:-}"; QW3_BIN="$2"; shift 2 ;;
    --prompt) need_arg "$1" "${2:-}"; PROMPT="$2"; shift 2 ;;
    --ctx) need_arg "$1" "${2:-}"; CTX="$2"; shift 2 ;;
    --threads) need_arg "$1" "${2:-}"; THREADS="$2"; shift 2 ;;
    --ngl) need_arg "$1" "${2:-}"; NGL="$2"; shift 2 ;;
    --batch) need_arg "$1" "${2:-}"; BATCH="$2"; shift 2 ;;
    --decode-tokens) need_arg "$1" "${2:-}"; DECODE_TOKENS="$2"; shift 2 ;;
    --native-token-id) need_arg "$1" "${2:-}"; NATIVE_TOKEN_ID="$2"; shift 2 ;;
    --out-dir) need_arg "$1" "${2:-}"; OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "--model is required" >&2
  usage
  exit 1
fi

mkdir -p "${OUT_DIR}"

time_cmd() {
  local log="$1"
  shift
  /usr/bin/time -f 'elapsed=%e' "$@" >"${log}" 2>"${log}.time"
  awk -F= '/^elapsed=/{print $2}' "${log}.time"
}

calc_tok_s() {
  local tokens="$1"
  local elapsed="$2"
  awk -v t="${tokens}" -v e="${elapsed}" 'BEGIN{if (e <= 0) {print "0.000"} else {printf "%.3f", t/e}}'
}

calc_ms_per_tok() {
  local tok_s="$1"
  awk -v s="${tok_s}" 'BEGIN{if (s <= 0) {print "0.000"} else {printf "%.3f", 1000.0/s}}'
}

run_qw3_native_once() {
  local linear_backend="$1"
  local log="$2"
  time_cmd "${log}" \
    "${QW3_BIN}" \
    --backend qwen-native \
    --native-heavy \
    --native-kernels cuda \
    --native-linear-backend "${linear_backend}" \
    --native-token-id "${NATIVE_TOKEN_ID}" \
    --model "${MODEL}" \
    -p "${PROMPT}" \
    -n 1 >/dev/null
}

run_qw3_native_many() {
  local linear_backend="$1"
  local n="$2"
  local log_prefix="$3"
  local t0
  t0=$(date +%s.%N)
  local i
  for ((i=0; i<n; ++i)); do
    local token_id=$((NATIVE_TOKEN_ID + i))
    "${QW3_BIN}" \
      --backend qwen-native \
      --native-heavy \
      --native-kernels cuda \
      --native-linear-backend "${linear_backend}" \
      --native-token-id "${token_id}" \
      --model "${MODEL}" \
      -p "${PROMPT}" \
      -n 1 >"${log_prefix}.${i}.log" 2>"${log_prefix}.${i}.err"
  done
  local t1
  t1=$(date +%s.%N)
  awk -v a="${t0}" -v b="${t1}" 'BEGIN{printf "%.6f", (b-a)}'
}

run_qw3_llama() {
  local n="$1"
  local log="$2"
  time_cmd "${log}" \
    "${QW3_BIN}" \
    --backend llama-cli \
    --llama-cli "${LLAMA_CLI}" \
    --model "${MODEL}" \
    -p "${PROMPT}" \
    -n "${n}" \
    -c "${CTX}" \
    -t "${THREADS}" \
    -ngl "${NGL}" \
    -b "${BATCH}" >/dev/null
}

echo "Running benchmark..."
echo "model=${MODEL}"
echo "prompt=${PROMPT}"
echo "decode_tokens=${DECODE_TOKENS}"
echo

native_cublas_ttft=$(run_qw3_native_once cublas "${OUT_DIR}/native_cublas_ttft.log")
native_auto_ttft=$(run_qw3_native_once auto "${OUT_DIR}/native_auto_ttft.log")
llama_ttft=$(run_qw3_llama 1 "${OUT_DIR}/llama_ttft.log")

native_cublas_decode_elapsed=$(run_qw3_native_many cublas "${DECODE_TOKENS}" "${OUT_DIR}/native_cublas_decode")
native_auto_decode_elapsed=$(run_qw3_native_many auto "${DECODE_TOKENS}" "${OUT_DIR}/native_auto_decode")
llama_decode_elapsed=$(run_qw3_llama "${DECODE_TOKENS}" "${OUT_DIR}/llama_decode.log")

native_cublas_tps=$(calc_tok_s "${DECODE_TOKENS}" "${native_cublas_decode_elapsed}")
native_auto_tps=$(calc_tok_s "${DECODE_TOKENS}" "${native_auto_decode_elapsed}")
llama_tps=$(calc_tok_s "${DECODE_TOKENS}" "${llama_decode_elapsed}")

native_cublas_ms=$(calc_ms_per_tok "${native_cublas_tps}")
native_auto_ms=$(calc_ms_per_tok "${native_auto_tps}")
llama_ms=$(calc_ms_per_tok "${llama_tps}")

cat >"${OUT_DIR}/summary.txt" <<EOF
Benchmark Summary
=================
model=${MODEL}
prompt=${PROMPT}
decode_tokens=${DECODE_TOKENS}

| backend | TTFT (s) | decode tok/s | decode latency (ms/token) |
|---|---:|---:|---:|
| qwen-native(cublas) | ${native_cublas_ttft} | ${native_cublas_tps} | ${native_cublas_ms} |
| qwen-native(auto)   | ${native_auto_ttft} | ${native_auto_tps} | ${native_auto_ms} |
| llama.cpp           | ${llama_ttft} | ${llama_tps} | ${llama_ms} |
EOF

cat "${OUT_DIR}/summary.txt"
echo
echo "raw logs: ${OUT_DIR}"
