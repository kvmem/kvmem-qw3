#!/usr/bin/env bash
# Full fp16-vs-fp8 KV-cache sweep: efficiency (prefill+decode tok/s) + resident
# KV memory + utility (passkey) across input sizes, plus GSM8K. Runs ONE server
# per dtype sequentially at ctx=131072 (no memory contention; clean throughput).
#
# Usage: scripts/kv_fp8_full_sweep.sh [DTYPE] [PORT]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$ROOT/models/Qwen3.6-27B-Q8_0.gguf"
QW3="$ROOT/build/qw3"
PY="${PYTHON:-python3}"
DT="${1:-fp16}"
PORT="${2:-8090}"
CTX=131072
LENS="4000,16000,64000,128000"
LOG="/tmp/qw3_sweep_${DT}.log"
OUT="/tmp/qw3_sweep_${DT}.out"
: > "$OUT"

wait_health() {
    for ((i=0;i<400;i++)); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "healthy after ${i}s"; return 0; }
        sleep 1
    done
    echo "server never healthy" >&2; return 1
}

gpu_mem() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep "^$1," | sed 's/.*, //'; }

# Build the request JSON entirely inside python (avoids passing a huge prompt
# as argv -> "Argument list too long"). Writes payload to $1; $2=ntokens $3=maxtok
build_payload() {
    "$PY" - "$1" "$2" "$3" <<'PYEOF'
import json, sys
path, ntok, maxtok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
prompt = "The quick brown fox jumps over the lazy dog. " * (ntok // 10)
with open(path, "w") as f:
    json.dump({"model": "m",
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": maxtok}, f)
PYEOF
}

req() {
    local ntok="$1" maxtok="$2"
    local pf="/tmp/qw3_sweep_payload.json"
    build_payload "$pf" "$ntok" "$maxtok"
    curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$pf" >/dev/null 2>&1 || true
}

echo "==== dtype=$DT port=$PORT ctx=$CTX ====" | tee -a "$OUT"
"$QW3" serve --model "$MODEL" --backend qwen-native -c "$CTX" --kv-dtype "$DT" --port "$PORT" > "$LOG" 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT
wait_health

echo "--- warmup (allocates full KV at ctx=$CTX) ---" | tee -a "$OUT"
req 4000 8
echo "resident_after_warmup_4k=$(gpu_mem "$PID")" | tee -a "$OUT"

echo "--- EFFICIENCY (prefill+decode, 64 decode tok, greedy) ---" | tee -a "$OUT"
IFS=',' read -ra L <<< "$LENS"
for n in "${L[@]}"; do
    req "$n" 64
    mem="$(gpu_mem "$PID")"
    line="$(grep 'native generate' "$LOG" | tail -1)"
    echo "len~$n | $line | resident=$mem" | tee -a "$OUT"
done

echo "--- PASSKEY UTILITY (lens=$LENS depths=0.1,0.5,0.9 trials=3 greedy) ---" | tee -a "$OUT"
"$PY" "$ROOT/scripts/kv_q8_utility.py" --base-url "http://127.0.0.1:$PORT/v1" \
    --lens "$LENS" --depths 0.1,0.5,0.9 --trials 3 2>&1 | tee -a "$OUT"

echo "--- GSM8K n=40 (greedy) ---" | tee -a "$OUT"
"$PY" "$ROOT/scripts/kv_q8_gsm8k.py" --base-url "http://127.0.0.1:$PORT/v1" --n 40 2>&1 | tail -6 | tee -a "$OUT"

kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
trap - EXIT
echo "==== done $DT ====" | tee -a "$OUT"
