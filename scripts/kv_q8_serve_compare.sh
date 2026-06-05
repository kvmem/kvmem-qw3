#!/usr/bin/env bash
# fp16-vs-q8 KV-cache utility comparison through the persistent qw3 serve API.
#
# Replaces the old reload-per-request methodology: instead of spawning ./build/qw3
# (and reloading the 27B model) for every problem, we launch ONE server per KV
# dtype, run both eval suites (GSM8K + passkey) against it, then move to the next
# dtype. The model loads exactly twice total (once per dtype) instead of ~hundreds
# of times.
#
# Usage:
#   scripts/kv_q8_serve_compare.sh [MODEL_GGUF] [GSM8K_N]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}"
GSM_N="${2:-40}"
CTX=4096
QW3="$ROOT/build/qw3"
PYTHON="${PYTHON:-python3}"

wait_health() {
    local url="$1"; local tries=180
    for ((i=0; i<tries; i++)); do
        if curl -sf "$url/health" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "server at $url never became healthy" >&2
    return 1
}

run_dtype() {
    local dt="$1"; local port="$2"
    local base="http://127.0.0.1:$port/v1"
    local log="/tmp/qw3_serve_${dt}.log"
    echo "=============================================================="
    echo "  KV dtype: $dt   (port $port)"
    echo "=============================================================="
    "$QW3" serve --model "$MODEL" -c "$CTX" --kv-dtype "$dt" \
        --port "$port" --prefill-chunk 1 >"$log" 2>&1 &
    local pid=$!
    trap "kill $pid 2>/dev/null || true" RETURN
    wait_health "http://127.0.0.1:$port"
    echo "--- GSM8K ($dt) ---"
    "$PYTHON" "$ROOT/scripts/kv_q8_gsm8k.py" --base-url "$base" --n "$GSM_N"
    echo "--- Passkey ($dt) ---"
    "$PYTHON" "$ROOT/scripts/kv_q8_utility.py" --base-url "$base"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    trap - RETURN
}

echo "model=$MODEL  ctx=$CTX  gsm8k_n=$GSM_N"
run_dtype fp16 8080
run_dtype q8 8081
echo
echo "Done. Compare the two GSM8K / Passkey accuracy blocks above for the"
echo "fp16-vs-q8 utility delta (server-side dtype, one model load per dtype)."
