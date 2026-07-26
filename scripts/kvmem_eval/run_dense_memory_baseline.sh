#!/usr/bin/env bash
# Controlled GPU-memory baseline for the same active-token capacity used by the
# 200K-context + 32K-generation KVMem configuration. This intentionally runs
# without KVMem and submits an exact integer-token prompt so tokenization and
# prompt contents cannot change the allocation size.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8088}
TAG=${TAG:-dense_fp16_mtp4_232k_memory_baseline_$(date +%Y%m%d_%H%M%S)}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
CTX=${CTX:-262144}
PROMPT_TOKENS=${PROMPT_TOKENS:-235520}
KV_DTYPE=${KV_DTYPE:-fp16}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
RESULT_ROOT=${RESULT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
GPU_LOG=${GPU_LOG:-$LOG_ROOT/${TAG}_gpu.csv}
SUMMARY=${SUMMARY:-$RESULT_ROOT/memory_summary.json}

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
server_pid=""
monitor_pid=""
cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  echo "timestamp,pid,used_memory_mib"
  while true; do
    timestamp=$(LC_ALL=C date --iso-8601=ns)
    nvidia-smi --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits 2>/dev/null |
      while IFS= read -r row; do
        echo "$timestamp,$row"
      done
    sleep 0.2
  done
) >"$GPU_LOG" 2>&1 &
monitor_pid=$!

env \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx "$CTX" --kv-dtype "$KV_DTYPE" \
    --enable-thinking --thinking-budget 4096 \
    --prefill-chunk "$PREFILL_CHUNK" --temp 0 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

healthy=0
for _ in $(seq 1 300); do
  if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2 || true
    exit 4
  fi
  sleep 2
done
if [[ "$healthy" -ne 1 ]]; then
  echo "dense memory baseline health timeout" >&2
  exit 4
fi

idle_mib=$(
  nvidia-smi --query-compute-apps=pid,used_memory \
    --format=csv,noheader,nounits |
  awk -F',' -v pid="$server_pid" '
    {
      gsub(/[[:space:]]/, "", $1)
      gsub(/[[:space:]]/, "", $2)
      if ($1 == pid) {
        print $2
        exit
      }
    }
  '
)
echo "[gpu-memory] phase=server_ready pid=$server_pid used_mib=$idle_mib"

"$ROOT/.venv/bin/python" - \
  "$PORT" "$PROMPT_TOKENS" "$RESULT_ROOT/completion.json" <<'PY'
import json
import pathlib
import sys
import time
import urllib.request

port = int(sys.argv[1])
prompt_tokens = int(sys.argv[2])
output = pathlib.Path(sys.argv[3])
url = f"http://127.0.0.1:{port}/v1/completions"
# Token 64 is a valid ordinary vocabulary token for this tokenizer. The
# request is a memory-capacity control, not a quality measurement.
payload = {
    "model": "Qwen3.6-27B-Q8_0.gguf",
    "prompt": [64] * prompt_tokens,
    "temperature": 0,
    "max_tokens": 1,
    "stream": False,
}
request = urllib.request.Request(
    url,
    data=json.dumps(payload, separators=(",", ":")).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
t0 = time.monotonic()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=1800) as response:
    result = json.loads(response.read())
result["client_elapsed_sec"] = time.monotonic() - t0
output.write_text(json.dumps(result, indent=2) + "\n")
print(
    f"[dense-memory] prompt_tokens={prompt_tokens} "
    f"elapsed_sec={result['client_elapsed_sec']:.3f}"
)
PY

sleep 2
post_request_mib=$(
  nvidia-smi --query-compute-apps=pid,used_memory \
    --format=csv,noheader,nounits |
  awk -F',' -v pid="$server_pid" '
    {
      gsub(/[[:space:]]/, "", $1)
      gsub(/[[:space:]]/, "", $2)
      if ($1 == pid) {
        print $2
        exit
      }
    }
  '
)

"$ROOT/.venv/bin/python" - \
  "$GPU_LOG" "$server_pid" "$idle_mib" "$post_request_mib" \
  "$PROMPT_TOKENS" "$KV_DTYPE" "$PREFILL_CHUNK" "$SUMMARY" <<'PY'
import csv
import json
import pathlib
import sys

gpu_log = pathlib.Path(sys.argv[1])
pid = int(sys.argv[2])
idle = int(sys.argv[3])
post = int(sys.argv[4])
prompt_tokens = int(sys.argv[5])
kv_dtype = sys.argv[6]
prefill_chunk = int(sys.argv[7])
summary_path = pathlib.Path(sys.argv[8])
samples = []
with gpu_log.open() as handle:
    for row in csv.reader(handle):
        if len(row) < 3 or row[1].strip() == "pid":
            continue
        try:
            # GNU date's nanosecond timestamp uses a decimal comma under the
            # current locale, so take the final two fields rather than fixed
            # timestamp-relative indices.
            row_pid = int(row[-2].strip())
            used = int(row[-1].strip())
        except ValueError:
            continue
        if row_pid == pid:
            samples.append(used)
summary = {
    "mode": "dense_no_kvmem",
    "pid": pid,
    "prompt_tokens": prompt_tokens,
    "kv_dtype": kv_dtype,
    "prefill_chunk": prefill_chunk,
    "server_ready_gpu_mib": idle,
    "post_request_gpu_mib": post,
    "peak_gpu_mib": max(samples) if samples else None,
    "gpu_samples": len(samples),
    "gpu_log": str(gpu_log),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
