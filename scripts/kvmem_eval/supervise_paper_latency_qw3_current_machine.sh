#!/usr/bin/env bash
# Controlled current-machine TTFT samples for the plain-qw3 baselines.
#
# The script is resumable: RAG preparation, strict compaction checkpoints and
# answer files are reused only when their frozen config hashes match.  It uses
# one server request at a time and max_tokens=1 so answer decoding after the
# first token cannot contaminate the paper metric.
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
# Keep the embedding environment separate from serving.  All model execution
# in the formal latency pass goes through this repository's qw3 binary.
RUN_ENV=${RUN_ENV:-/home/chaidi/qw3/.venv}
RUN_PY=${RUN_PY:-$RUN_ENV/bin/python}
QW3_BIN=${QW3_BIN:-$ROOT/build/qw3}
MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
MODEL_NAME=${MODEL_NAME:-Qwen3.6-27B-Q8_0.gguf}
TOKENIZER=${TOKENIZER:-/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8}
RAG_MODEL=${RAG_MODEL:-/data/chaidi/kvmem_eval/models/jinaai/jina-embeddings-v2-small-en}
OUT=${OUT:-/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812}
PORT=${PORT:-18124}
API_BASE="http://127.0.0.1:${PORT}/v1"

ALB512_RUNNER=/home/chaidi/AgentLongBench-Long/script/compactRag/run_no_tail_cap64k.py
ALB1M_RUNNER=/home/chaidi/AgentLongBench-Long/script/deepseekMillion/run_no_tail_cap100k_1m.py
ALB_WINDOW=/home/chaidi/AgentLongBench-Long/script/slidingWindow/run_sliding_window.py
ALB_COMPACT_ONLY=$ROOT/scripts/kvmem_eval/measure_agentlongbench_strict_compact_only.py
MAB_RUNNER=$ROOT/scripts/kvmem_eval/run_memoryagentbench_baselines.py
ALB_LE_FULL=$ROOT/scripts/kvmem_eval/measure_agentlongbench_full_cached_latency.py
ALB_LE_SLIDING=/home/chaidi/AgentLongBench_Motivation/slidingwindow/run_agentlongbench_sliding_window_worker.py
ALB_LE_SUMMARY=/home/chaidi/AgentLongBench_Motivation/compactonly/run_agentlongbench_summary_worker.py
ALB_LE_COMPACT=/home/chaidi/AgentLongBench_Motivation/compactrag/run_agentlongbench_compact_eval_worker.py
LME_SLIDING=$ROOT/scripts/kvmem_eval/measure_longmemeval_sliding_latency.py
LME_ROOT=/home/chaidi/kvmem_eval/KVMem_Motivation
LME_FULL_PROBE=$LME_ROOT/cache_hit_timing/run_cache_hit_smoke.py
LME_COMPACT_PROBE=$LME_ROOT/cache_hit_timing/run_compact_rag_cache_hit_smoke.py
LME_SELECTION=$ROOT/scripts/kvmem_eval/paper_latency_lme_s_samples_20260812.jsonl
LME_FULL_CONFIG=$ROOT/scripts/kvmem_eval/paper_latency_lme_s_full_current.json
LME_COMPACT_CONFIG=$ROOT/scripts/kvmem_eval/paper_latency_lme_s_compact_current.json
LME_RAG_CONFIG=$ROOT/scripts/kvmem_eval/paper_latency_lme_s_rag_k30_current.json

DATA512=/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl
DATA1M=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl
SOURCE512_CURRENT=$OUT/alb512_compaction_current
SOURCE1M=$OUT/alb1m_compaction_current
MAB_REFERENCE=/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full
DATA_LE=/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl
MANIFEST_LE=/data/chaidi/kvmem_eval/data/agentlongbench_250/manifest.jsonl
ALB_LE_REPO=/home/chaidi/AgentLongBench_Motivation
LME_DATA=/data/chaidi/kvmem_eval/data/longmemeval_s.json

IDS512=(
  346db45888398d0cd09cf20734b719bc2d5cf6243a530727e1b9ac20c2afed78
)
IDS1M=(
  5052ef703e33f939469245773a16570d87729aba959421457045fbf12fcbf73e
)
IDS_LE=(
  febc5e457e02ecea1bc51e9f3955591ee48e573c44e15a435d15c0c474062fcd
)
IDS_LME=(1d4e3b97)

mkdir -p "$OUT/logs"
exec >>"$OUT/logs/supervisor.log" 2>&1
echo "[$(date --iso-8601=seconds)] start current-machine plain-qw3 latency samples"

env OUT="$OUT" QW3_BIN="$QW3_BIN" MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" TOKENIZER="$TOKENIZER" \
  python3 - <<'PY'
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import hashlib
import sys

def command(*argv: str) -> str:
    return subprocess.check_output(argv, text=True).strip()

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
    "backend": "qw3",
    "python": sys.executable,
    "qw3_binary": str(Path(os.environ["QW3_BIN"]).resolve()),
    "qw3_binary_sha256": hashlib.sha256(Path(os.environ["QW3_BIN"]).read_bytes()).hexdigest(),
    "qw3_binary_size": Path(os.environ["QW3_BIN"]).stat().st_size,
    "model_path": str(Path(os.environ["MODEL_PATH"]).resolve()),
    "model_name": os.environ["MODEL_NAME"],
    "tokenizer_path": str(Path(os.environ["TOKENIZER"]).resolve()),
    "server_contract": {
        "kv_cache_dtype": "fp8",
        "max_model_len": 262_144,
        "max_num_seqs": 1,
        "prefill_chunk": 2_048,
        "chunked_prefill": True,
        "prefix_caching": True,
        # Native qw3 prefix-cache v1 requires a non-MTP plain server.  MTP is
        # disabled only for these dense resident-prefix latency probes.
        "mtp_speculative_tokens": 0,
    },
}
path = Path(os.environ["OUT"]) / "environment_qw3_plain.json"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY

qid_args() {
  local value
  for value in "$@"; do
    printf '%s\n' --question-id "$value"
  done
}
mapfile -t Q512 < <(qid_args "${IDS512[@]}")
mapfile -t Q1M < <(qid_args "${IDS1M[@]}")
mapfile -t QLE < <(qid_args "${IDS_LE[@]}")

ALB_COMMON=(
  --model "$MODEL_NAME"
  --tokenizer-path "$TOKENIZER"
  --embedding-model-path "$RAG_MODEL"
  --rag-chunk-tokenizer-path "$TOKENIZER"
  --embedding-device cuda
  --embedding-batch-size 256
  --api-backend qw3
  --workers 1
  --max-tokens 1
  --attempts 2
  --timeout-sec 14400
  --max-call-sec 14400
)

# Build the query-specific RAG indexes before qw3 reserves the GPU.
env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" "$ALB512_RUNNER" \
    --dataset "$DATA512" --source-compact-root "$SOURCE512_CURRENT" \
    --output-root "$OUT/alb512_compact_rag" \
    "${ALB_COMMON[@]}" "${Q512[@]}" --prepare-rag-only

env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" "$ALB1M_RUNNER" \
    --dataset "$DATA1M" --source-compact-root "$SOURCE1M" \
    --output-root "$OUT/alb1m_compact_rag" \
    "${ALB_COMMON[@]}" "${Q1M[@]}" --prepare-rag-only

MAB_BASE=(
  --reference-results "$MAB_REFERENCE"
  --min-context-tokens-exclusive 262144
  --context-window 262144
  --generation-reserve 32768
  --strict-final-prompt-tokens 65536
  --sliding-prompt-tokens 65536
  --compact-full-history
  --compact-no-tail
  --rag-dynamic-budget
  --rag-no-metadata
  --rag-top-k 2200
  --rag-block-size 32
  --rag-overlap 8
  --rag-model "$RAG_MODEL"
  --tokenizer-dir "$TOKENIZER"
  --embedding-batch-size 256
  --question-limit 1
  --answer-max-tokens-override 1
)
MAB_SPECS=(
  'mab_p50|Long_Range_Understanding|infbench_sum_eng_shots2|10'
)
for spec in "${MAB_SPECS[@]}"; do
  IFS='|' read -r label split source row <<<"$spec"
  workspace="$OUT/${label}_boundary_v2"
  env PYTHONUNBUFFERED=1 "$PY" "$MAB_RUNNER" prepare \
    --workspace "$workspace" "${MAB_BASE[@]}" \
    --split "$split" --source "$source" --row "$row"
  env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PY" "$MAB_RUNNER" prepare-rag \
      --workspace "$workspace" "${MAB_BASE[@]}" \
      --split "$split" --source "$source" --row "$row" \
      --prepare-rag-respect-question-limit --embedding-device cuda
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

server_run=0
start_server() {
  server_run=$((server_run + 1))
  local server_log="$OUT/logs/server_qw3_plain_fp8_nomtp_seq1_run${server_run}.log"
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    QW3_Q8_BF16_MAIN=0 QW3_FATTN_NSPLIT=1 QW3_PREFILL_FA2_NSPLIT=1 \
    QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 QW3_PREFIX_CACHE_TRACE=1 \
    QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES=1 QW3_PREFIX_CACHE_MAX_PAGES=8192 \
    "$QW3_BIN" serve --model "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" \
      --ctx 262144 --kv-dtype fp8 --prefill-chunk 2048 \
      --continuous-batching --prefix-cache --max-active 1 --max-pending 8 \
      >"$server_log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 240); do
    kill -0 "$server_pid" 2>/dev/null || {
      tail -120 "$server_log"; exit 1;
    }
    curl --noproxy '*' -fsS "$API_BASE/models" >/dev/null && break
    sleep 2
  done
  curl --noproxy '*' -fsS "$API_BASE/models" >/dev/null
}

restart_server() {
  stop_server
  start_server
}

start_server

# Unrelated warm-up: it cannot warm any benchmark prefix or retrieval.
curl --noproxy '*' -fsS "$API_BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK.\"}],\"max_tokens\":1,\"stream\":false,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  >"$OUT/logs/warmup.json"

# The short request above covers decode/MTP, but it does not compile or tune
# the chunked-prefill kernels used by a 64K/100K cold window.  Without this
# unrelated long warm-up, the first measured Sliding Window sample absorbs
# one-time Triton JIT work while every later method gets a warm kernel.  Use
# synthetic text that cannot share a benchmark prefix and exercise both paper
# prompt caps before any recorded request.
env API_BASE="$API_BASE" MODEL_NAME="$MODEL_NAME" OUT="$OUT" \
  "$RUN_PY" - <<'PY'
import json
import os
from pathlib import Path
import time

import requests

api_base = os.environ["API_BASE"].rstrip("/")
model = os.environ["MODEL_NAME"]
out = Path(os.environ["OUT"]) / "logs" / "warmup_long.json"
rows = []
session = requests.Session()
session.trust_env = False
for target in (65_536, 102_400):
    # This tokenizer maps ``" x"`` to one token.  Chat-template overhead is
    # only a few tokens, so these requests closely reproduce the two active
    # prompt lengths while remaining safely below max_model_len.
    content = (" x" * target).strip()
    started = time.perf_counter()
    response = session.post(
        api_base + "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=3600,
    )
    response.raise_for_status()
    payload = response.json()
    rows.append({
        "synthetic_repetitions": target,
        "elapsed_sec": time.perf_counter() - started,
        "usage": payload.get("usage"),
        "timing": payload.get("timing"),
    })
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
PY

WINDOW_COMMON=(
  --api-base "$API_BASE" --model "$MODEL_NAME" --api-backend qw3
  --tokenizer-path "$TOKENIZER" --temperature 0.6 --top-p 0.95 --top-k 20
  --thinking-budget 8192 --answer-max-tokens 1 --chat-template-reserve 16
  --seed 20260722 --attempts 2 --timeout-sec 14400 --max-call-sec 14400
)
# These prompt-only baselines do not need the Jina environment.  Use the exact
# serving tokenizer build so the selected cold window and server-side cap are
# measured under one tokenizer implementation.
env PYTHONUNBUFFERED=1 "$RUN_PY" "$ALB_WINDOW" run \
  --dataset-path "$DATA512" --output-root "$OUT/alb512_sliding_boundary_v2" \
  --window-prompt-tokens 65536 --latency-warm-prefix \
  "${WINDOW_COMMON[@]}" "${Q512[@]}"
env PYTHONUNBUFFERED=1 "$RUN_PY" "$ALB_WINDOW" run \
  --dataset-path "$DATA1M" --output-root "$OUT/alb1m_sliding_boundary_v2" \
  --window-prompt-tokens 102400 --latency-warm-prefix \
  "${WINDOW_COMMON[@]}" "${Q1M[@]}"

# Reuse the offline retrievals, but regenerate the query-boundary compaction
# and final prompt in this isolated run.
# First construct the two question-blind source compaction rounds on this same
# server.  This calls the existing compact-only implementation directly while
# using the strict runner's vLLM client, so no additional public CLI mode is
# needed and the configured 10K thinking budget remains enforced.
env PYTHONUNBUFFERED=1 DATA512="$DATA512" SOURCE512_CURRENT="$SOURCE512_CURRENT" \
  API_BASE="$API_BASE" MODEL_NAME="$MODEL_NAME" TOKENIZER="$TOKENIZER" \
  IDS512_JOINED="${IDS512[*]}" "$RUN_PY" - <<'PY'
import argparse
import importlib.util
import os
from pathlib import Path

runner_path = Path("/home/chaidi/AgentLongBench-Long/script/compactRag/run_no_tail_cap64k.py")
spec = importlib.util.spec_from_file_location("alb512_strict_latency", runner_path)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)
compact = runner.compact
samples = compact.load_samples(Path(os.environ["DATA512"]))
wanted = set(os.environ["IDS512_JOINED"].split())
selected = [sample for sample in samples if compact.sample_id(sample) in wanted]
if {compact.sample_id(sample) for sample in selected} != wanted:
    raise SystemExit("ALB512 current-machine compaction sample selection mismatch")
root = Path(os.environ["SOURCE512_CURRENT"])
client = runner.ControlledApiClient(
    os.environ["API_BASE"],
    os.environ["MODEL_NAME"],
    "dummy",
    14_400,
    14_400,
    os.environ["TOKENIZER"],
    "Qwen2TokenizerFast-pinned",
    temperature=0.0,
    top_p=0.9,
    thinking_budget=10_000,
    api_backend="qw3",
)
args = argparse.Namespace(disable_thinking=False, attempts=2, save_reasoning=False)
for index, sample in enumerate(selected, 1):
    compact.process_compaction_only(sample, index, len(selected), root, client, args)
print(f"[alb512-current-compaction] completed={len(selected)}", flush=True)
PY

env PYTHONUNBUFFERED=1 "$PY" "$ALB512_RUNNER" \
  --dataset "$DATA512" --source-compact-root "$SOURCE512_CURRENT" \
  --output-root "$OUT/alb512_compact_rag" --api-base "$API_BASE" \
  "${ALB_COMMON[@]}" "${Q512[@]}"
env PYTHONUNBUFFERED=1 "$PY" "$ALB1M_RUNNER" \
  --dataset "$DATA1M" --source-compact-root "$SOURCE1M" \
  --output-root "$OUT/alb1m_compact_rag" --api-base "$API_BASE" \
  "${ALB_COMMON[@]}" "${Q1M[@]}"

env PYTHONUNBUFFERED=1 "$PY" "$ALB_COMPACT_ONLY" \
  --slice 512k --dataset "$DATA512" \
  --source-compact-rag-root "$OUT/alb512_compact_rag" \
  --output "$OUT/alb512_compact_only_boundary_v2.jsonl" --api-base "$API_BASE" \
  --model "$MODEL_NAME" --tokenizer-path "$TOKENIZER" \
  --max-tokens 1 "${Q512[@]}"
env PYTHONUNBUFFERED=1 "$PY" "$ALB_COMPACT_ONLY" \
  --slice 1m --dataset "$DATA1M" \
  --source-compact-rag-root "$OUT/alb1m_compact_rag" \
  --output "$OUT/alb1m_compact_only_boundary_v2.jsonl" --api-base "$API_BASE" \
  --model "$MODEL_NAME" --tokenizer-path "$TOKENIZER" \
  --max-tokens 1 "${Q1M[@]}"

# Do not pass --source-shared-workspace in the sampled latency run.  Utility
# may safely reuse its frozen, question-independent summaries, but doing that
# here would leave the parenthesized (steady-state) number current-machine
# measured while the main number silently inherited historical compaction
# timings.  The first compact-rag request below therefore builds the complete
# no-tail summary on this server; compact-only then reuses the same newly built
# state, so both methods are charged the identical measured compaction cost.
for spec in "${MAB_SPECS[@]}"; do
  IFS='|' read -r label split source row <<<"$spec"
  workspace="$OUT/${label}_boundary_v2"
  for method in compact-rag compact-only sliding-window; do
    env PYTHONUNBUFFERED=1 "$RUN_PY" "$MAB_RUNNER" run \
      --workspace "$workspace" "${MAB_BASE[@]}" \
      --split "$split" --source "$source" --row "$row" \
      --api-base "$API_BASE" --model-name "$MODEL_NAME" \
      --server-mtp-chain 0 --prefix-cache-guard-pages 1 \
      --method "$method"
  done
done

# AgentLongBench <=256K.  The ID is the deterministic P50 representative of
# the frozen 250-question population.
env PYTHONUNBUFFERED=1 "$RUN_PY" "$ALB_LE_FULL" \
  --dataset "$DATA_LE" --benchmark-repo "$ALB_LE_REPO" \
  "${QLE[@]}" --api-base "$API_BASE" --model "$MODEL_NAME" \
  --output "$OUT/alb_le256_full_cached_boundary_v2.jsonl"

# Prefix-cache entries are deliberately not shared across paper methods.  A
# 148K Full Context entry plus the next resident 32K window can otherwise
# exhaust the 256K physical page pool before eviction completes.  Restarting
# only the service clears method-local KV; model-load time remains outside the
# measured request latency.
restart_server
env PYTHONUNBUFFERED=1 "$RUN_PY" "$ALB_LE_SLIDING" \
  --dataset-path "$DATA_LE" --selected-manifest-path "$MANIFEST_LE" \
  --output-root "$OUT/alb_le256_sliding_boundary_v2" --api-base "$API_BASE" \
  --model "$MODEL_NAME" --temperature 0.7 --top-p 0.9 --max-tokens 1 \
  --sliding-window-tokens 32768 --sliding-safety-margin 1024 \
  --answer-attempts 1 --timeout-sec 3600 --max-sample-sec 3600 \
  "${QLE[@]}"

restart_server
env PYTHONUNBUFFERED=1 "$PY" "$ALB_LE_SUMMARY" \
  --dataset-path "$DATA_LE" --selected-manifest-path "$MANIFEST_LE" \
  --output-root "$OUT/alb_le256_summary_boundary_v2" --run-label paperlat_v2 \
  --api-base "$API_BASE" --model "$MODEL_NAME" \
  --temperature 0.0 --top-p 0.9 --summary-max-tokens 60000 \
  --summary-prompt-mode codex_compact --allow-over-budget-prompt \
  --enable-thinking --attempts 1 --timeout-sec 3600 --max-sample-sec 3600 \
  "${QLE[@]}"

restart_server
env PYTHONUNBUFFERED=1 "$PY" "$ALB_LE_COMPACT" \
  --dataset-path "$DATA_LE" --selected-manifest-path "$MANIFEST_LE" \
  --summary-path "$OUT/alb_le256_summary_boundary_v2/summaries/full_history_s_all_paperlat_v2.jsonl" \
  --output-root "$OUT/alb_le256_compact_eval_boundary_v2" --run-label paperlat_v2 \
  --api-base "$API_BASE" --model "$MODEL_NAME" \
  --temperature 0.0 --top-p 0.9 --answer-max-tokens 1 \
  --methods compact_only compact_rag_t30 \
  --rag-top-k 30 --rag-block-size 1024 --rag-chunk-overlap 128 \
  --rag-embedding-model all-MiniLM-L6-v2 --rag-embedding-device cpu \
  --rag-embedding-model-path /data/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --enable-thinking --answer-attempts 1 --timeout-sec 3600 \
  --max-sample-sec 3600 "${QLE[@]}"

# LongMemEval-S P50.  Full/Compact probes explicitly warm only the
# query-independent prefix.  Sliding remains a cold strict-32K prompt.
for qid in "${IDS_LME[@]}"; do
  restart_server
  env PYTHONUNBUFFERED=1 "$PY" "$LME_FULL_PROBE" \
    --config "$LME_FULL_CONFIG" --question-id "$qid" \
    --warmup-max-tokens 0 --answer-max-tokens 1 \
    --output-dir "$OUT/lme_s_full_context"
done

restart_server
env PYTHONUNBUFFERED=1 "$RUN_PY" "$LME_SLIDING" \
  --data "$LME_DATA" --selection "$LME_SELECTION" \
  --tokenizer "$TOKENIZER" --api-base "$API_BASE" --model "$MODEL_NAME" \
  --active-cap 32768 --question-id "${IDS_LME[0]}" \
  --output "$OUT/lme_s_sliding_boundary_v2.jsonl"

for qid in "${IDS_LME[@]}"; do
  restart_server
  env PYTHONUNBUFFERED=1 "$PY" "$LME_COMPACT_PROBE" \
    --compact-config "$LME_COMPACT_CONFIG" \
    --retrieval-config "$LME_RAG_CONFIG" \
    --selection-path "$LME_SELECTION" --question-id "$qid" \
    --summary-prefix-target-tokens 100000 --warmup-max-tokens 0 \
    --summary-max-tokens 32768 --answer-max-tokens 1 \
    --output-dir "$OUT/lme_s_compact_rag"
done

stop_server
echo "[$(date --iso-8601=seconds)] completed current-machine plain-qw3 latency samples"
