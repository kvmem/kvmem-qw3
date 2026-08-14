#!/usr/bin/env bash
# Current-machine KVMem pre-answer latency for every non-TBD paper slice.
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
BIN=${BIN:-$ROOT/build/qw3}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
OUT=${OUT:-/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812}
PORT=${PORT:-18125}
API_BASE="http://127.0.0.1:${PORT}/v1"
SERVER_LOG="$OUT/logs/kvmem_long_server.log"
MEASURE=$ROOT/scripts/kvmem_eval/measure_frozen_kvmem_latency.py
LME_MEASURE=$ROOT/scripts/kvmem_eval/measure_longmemeval_frozen_kvmem_latency.py
MAB_RUNNER=$ROOT/scripts/kvmem_eval/run_memoryagentbench_archive.py

DATA512=/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl
DATA1M=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl
BENCH_REPO=/home/chaidi/AgentLongBench_Motivation
MAB_SELECTION=$ROOT/scripts/kvmem_eval/memoryagentbench_latency_sample_p50_over256k.jsonl
DATA_LE=/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl
DATA_LME=/data/chaidi/kvmem_eval/data/longmemeval_s.json
LME_SELECTION=$ROOT/scripts/kvmem_eval/paper_latency_lme_s_sample_p50_20260812.jsonl
TOKENIZER=/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8

IDS512=(
  346db45888398d0cd09cf20734b719bc2d5cf6243a530727e1b9ac20c2afed78
)
IDS1M=(
  5052ef703e33f939469245773a16570d87729aba959421457045fbf12fcbf73e
)
IDS_LE=(
  febc5e457e02ecea1bc51e9f3955591ee48e573c44e15a435d15c0c474062fcd
)

mkdir -p "$OUT/logs"
exec >>"$OUT/logs/kvmem_long_supervisor.log" 2>&1
echo "[$(date --iso-8601=seconds)] start KVMem long-slice latency"

env OUT="$OUT" BIN="$BIN" MODEL="$MODEL" ROOT="$ROOT" \
  "$ROOT/.venv/bin/python" - <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

def command(*argv: str) -> str:
    return subprocess.check_output(argv, text=True).strip()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()

binary = Path(os.environ["BIN"]).resolve()
model = Path(os.environ["MODEL"]).resolve()
gpu = command(
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader,nounits",
).splitlines()[0]
payload = {
    "schema_version": 1,
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": platform.node(),
    "gpu": gpu,
    "python": sys.executable,
    "git_commit": command("git", "-C", os.environ["ROOT"], "rev-parse", "HEAD"),
    "binary_path": str(binary),
    "binary_size": binary.stat().st_size,
    "binary_sha256": sha256(binary),
    "model_path": str(model),
    "model_size": model.stat().st_size,
    "common_contract": {
        "generation_reserve": 32_768,
        "prefill_chunk": 2_048,
        "query_replay": True,
        "immutable_k": True,
        "immutable_refresh_tokens": 8,
        "stage_out": True,
        "stage_in": True,
        "pack": True,
        "mtp_chain": 4,
    },
}
path = Path(os.environ["OUT"]) / "environment_kvmem.json"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY

# MemoryAgentBench uses the same durable archive path as its utility run.  A
# one-token query records an explicit native-engine ttft_s callback timestamp;
# archive construction/attach remain outside the measured per-question row.
MAB_RESULT="$OUT/mab_kvmem_k64_b64/rows/001_Long_Range_Understanding_r010_infbench_sum_eng_shots2/results.jsonl"
MAB_SUMMARY="$OUT/mab_kvmem_k64_b64/rows/001_Long_Range_Understanding_r010_infbench_sum_eng_shots2/row_summary.json"
if "$ROOT/.venv/bin/python" - "$MAB_RESULT" "$MAB_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

result_path, summary_path = map(Path, sys.argv[1:])
if not result_path.is_file() or not summary_path.is_file():
    raise SystemExit(1)
rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
summary = json.loads(summary_path.read_text())
valid = (
    len(rows) == 1
    and isinstance(rows[0].get("ttft_s"), (int, float))
    and rows[0]["ttft_s"] > 0
    and summary.get("questions_completed") == 1
    and summary.get("scorer_fallbacks") == 0
)
raise SystemExit(0 if valid else 1)
PY
then
  echo "[resume] validated MemoryAgentBench KVMem latency result"
else
  env PYTHONUNBUFFERED=1 "$PY" "$MAB_RUNNER" \
    --binary "$BIN" --model "$MODEL" \
    --out-dir "$OUT/mab_kvmem_k64_b64" \
    --selection-manifest "$MAB_SELECTION" --question-limit 1 \
    --answer-max-tokens-override 1 \
    --archive-storage cpu-only --tmpfs-root /dev/shm --tmpfs-limit-gib 60 \
    --ssd-root /home/chaidi/kca/paper_latency_unused \
    --block-tokens 64 --budget 65536 --gen-budget 32768 \
    --sink-tokens 512 --recent-tokens 0 --gpu-memory-ratio 1.0 \
    --cpu-gb-tmpfs 48 --cpu-gb-ssd 48 --prefill-chunk 2048 \
    --prefill-window semantic_chunk \
    --prefill-semantic-start-tokens 65536 \
    --prefill-semantic-query-tokens 0 \
    --immutable-refresh-tokens 8 --index-placement gpu \
    --index-staging-mb 64
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

start_server() {
  local ctx=$1 budget=$2 block=$3 thinking=$4 log=$5
  local kv_dtype=$6 retrieval=$7 sink_tokens=$8 cpu_gb=$9
  : >"$log"
  env \
    -u QW3_KVMEM_REBUILT_STATE_DIR \
    -u QW3_KVMEM_QUERY_SCORE_TOKENS \
    -u QW3_KVMEM_ADAPTIVE_GPU_PACKED \
    QW3_KVMEM_RECOMPUTE_QUERY=1 \
    QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
    QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=8 \
    QW3_KVMEM_TRACE=1 QW3_KVMEM_TIMING=1 \
    QW3_Q8_BF16_MAIN=0 QW3_FATTN_NSPLIT=1 QW3_PREFILL_FA2_NSPLIT=1 \
    QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
    QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB=512 \
    "$BIN" serve --model "$MODEL" --ctx "$ctx" --kv-dtype "$kv_dtype" \
      --kvmem --kvmem-block-tokens "$block" \
      --kvmem-budget "$budget" --kvmem-prefill-budget "$budget" \
      --kvmem-gen-budget 32768 --kvmem-sink-tokens "$sink_tokens" \
      --kvmem-recent-tokens 0 --kvmem-method retrieval \
      --kvmem-retrieval-method "$retrieval" \
      --kvmem-index-placement gpu --kvmem-index-staging-mb 64 \
      --kvmem-adaptive-score-mode auto \
      --kvmem-adaptive-gain-1to2 0.10 --kvmem-adaptive-gain-2to4 0.06 \
      --kvmem-update-mode step --kvmem-query-conditioned --kvmem-immutable-k \
      --kvmem-gpu-memory-ratio 1.0 --kvmem-cpu-gb "$cpu_gb" \
      --no-kvmem-raw-k-nvme --kvmem-opt-stage-out on \
      --kvmem-opt-stage-in on --kvmem-opt-pack on \
      --enable-thinking --thinking-budget "$thinking" \
      --prefill-chunk 2048 --temp 0.6 \
      --native-mtp-speculate --mtp-chain 4 \
      --host 127.0.0.1 --port "$PORT" >"$log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 300); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
      return 0
    fi
    kill -0 "$server_pid" 2>/dev/null || {
      tail -120 "$log"; return 1;
    }
    sleep 2
  done
  return 1
}

qid_args() {
  local value
  for value in "$@"; do printf '%s\n' --question-id "$value"; done
}

LOG512="$OUT/logs/alb512_kvmem_k64_b32_server.log"
start_server 655360 65536 32 8192 "$LOG512" fp8 key-direction-adaptive 512 48
mapfile -t Q512 < <(qid_args "${IDS512[@]}")
"$ROOT/.venv/bin/python" "$MEASURE" \
  --dataset "$DATA512" --benchmark-repo "$BENCH_REPO" \
  "${Q512[@]}" --slice alb_512k_k64_b32 \
  --api-base "$API_BASE" --model "$(basename "$MODEL")" \
  --server-log "$LOG512" --output "$OUT/alb512_kvmem.jsonl" \
  --active-context-budget 65536 --generation-reserve 32768 \
  --block-tokens 32 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --enable-thinking --thinking-budget 8192 \
  --prefill-window semantic_chunk \
  --prefill-semantic-start-tokens 65536 \
  --prefill-semantic-query-tokens 0
stop_server

LOG1M="$OUT/logs/alb1m_kvmem_k100_b128_server.log"
start_server 1310720 102400 128 8192 "$LOG1M" fp8 key-direction-adaptive 512 48
mapfile -t Q1M < <(qid_args "${IDS1M[@]}")
"$ROOT/.venv/bin/python" "$MEASURE" \
  --dataset "$DATA1M" --benchmark-repo "$BENCH_REPO" \
  "${Q1M[@]}" --slice alb_1m_k100_b128 \
  --api-base "$API_BASE" --model "$(basename "$MODEL")" \
  --server-log "$LOG1M" --output "$OUT/alb1m_kvmem.jsonl" \
  --active-context-budget 102400 --generation-reserve 32768 \
  --block-tokens 128 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --enable-thinking --thinking-budget 8192 \
  --prefill-window semantic_chunk \
  --prefill-semantic-start-tokens 102400 \
  --prefill-semantic-query-tokens 0
stop_server

# LongMemEval-S uses the frozen 32K adaptive utility configuration.  History
# ingest is excluded; the recorded request begins at the final semantic
# reselection and includes query replay through the first generated token.
LOGLME="$OUT/logs/longmemeval_s_kvmem_k32_b32_server.log"
start_server 262144 32768 32 4096 "$LOGLME" fp16 key-direction-adaptive 256 64
"$ROOT/.venv/bin/python" "$LME_MEASURE" \
  --data "$DATA_LME" --selection "$LME_SELECTION" --tokenizer "$TOKENIZER" \
  --api-base "$API_BASE" --model "$(basename "$MODEL")" \
  --server-log "$LOGLME" --output "$OUT/longmemeval_s_kvmem.jsonl" \
  --active-context-budget 32768 --generation-reserve 32768 \
  --block-tokens 32
stop_server

# AgentLongBench <=256K uses its published K32/B32 Mean-K configuration; only
# the P50 representative is replayed for the latency cell.
LOGLE="$OUT/logs/alb_le256_kvmem_k32_b32_server.log"
start_server 262144 32768 32 4096 "$LOGLE" fp16 mean-k 256 64
mapfile -t QLE < <(qid_args "${IDS_LE[@]}")
"$ROOT/.venv/bin/python" "$MEASURE" \
  --dataset "$DATA_LE" --benchmark-repo "$BENCH_REPO" \
  "${QLE[@]}" --slice alb_le256_k32_b32 \
  --api-base "$API_BASE" --model "$(basename "$MODEL")" \
  --server-log "$LOGLE" --output "$OUT/alb_le256_kvmem.jsonl" \
  --active-context-budget 32768 --generation-reserve 32768 \
  --block-tokens 32 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --enable-thinking --thinking-budget 4096 \
  --prefill-window pressure --final-reselect force
stop_server

python3 "$ROOT/scripts/kvmem_eval/validate_paper_latency_current.py" \
  --root "$OUT" --stage kvmem --output "$OUT/kvmem_validation.json"
echo "[$(date --iso-8601=seconds)] completed all KVMem paper latency slices"
