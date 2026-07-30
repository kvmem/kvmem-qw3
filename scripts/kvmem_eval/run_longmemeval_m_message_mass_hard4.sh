#!/usr/bin/env bash
# Controlled LongMemEval-M probe on the four samples that remained wrong in the
# prior 6/10 query-replay + immutable-K result. The standard flattened prompt is
# unchanged; the runner only supplies original-turn byte spans for complete
# message expansion. Set INDICES to run a different frozen subset.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
TAG=${TAG:-longmemeval_m_message_mass_alpha05_hard4_${STAMP}}
PORT=${PORT:-8089}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/longmemeval_m_102.jsonl}
INDICES=${INDICES:-6,34,60,86}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
RESULTS=${RESULTS:-/data/chaidi/kvmem_eval/results}
LOGS=${LOGS:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${SERVER_LOG:-$LOGS/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOGS/${TAG}_runner.log}
MANIFEST=${MANIFEST:-$RESULTS/${TAG}_manifest.json}
NVME_DIR=${NVME_DIR:-/home/chaidi/.cache/qw3-kvmem/$TAG}
CPU_GB=${CPU_GB:-64}
NVME_GB=${NVME_GB:-128}

mkdir -p "$RESULTS" "$LOGS" "$NVME_DIR"
if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; refusing to disturb it" >&2
  exit 3
fi

server_cmd=(
  "$ROOT/build/qw3" serve
  --model "$MODEL"
  --ctx 2000000
  --kv-dtype fp16
  --kvmem
  --kvmem-block-tokens 32
  --kvmem-budget 229376
  --kvmem-gen-budget 32768
  --kvmem-sink-blocks 8
  --kvmem-recent-blocks 0
  --kvmem-method retrieval
  --kvmem-retrieval-method sub-block-mean-k
  --kvmem-subblocks 1
  --kvmem-subblock-reduce max
  --kvmem-semantic-expansion message
  --kvmem-group-score-reduce length-normalized-mass
  --kvmem-group-length-alpha 0.5
  --kvmem-update-mode step
  --kvmem-query-conditioned
  --kvmem-immutable-k
  --kvmem-gpu-memory-ratio 0.5
  --kvmem-cpu-gb "$CPU_GB"
  --kvmem-nvme-gb "$NVME_GB"
  --kvmem-nvme-dir "$NVME_DIR"
  --enable-thinking
  --thinking-budget 4096
  --prefill-chunk 2048
  --temp 0.6
  --native-mtp-speculate
  --mtp-chain 4
  --host 127.0.0.1
  --port "$PORT"
)

judge_args=()
if [[ -z ${DEEPSEEK_API_KEY:-} ]]; then
  judge_args+=(--no-judge)
  echo "DEEPSEEK_API_KEY is not set; generation will run now and grading can be attached later"
fi

eval_cmd=(
  "$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_eval.py"
  --data "$DATA"
  --use-all
  --base-url "http://127.0.0.1:$PORT/v1"
  --out-dir "$RESULTS"
  --tag "$TAG"
  --model "$(basename "$MODEL")"
  --max-tokens 32768
  --temperature 0.6
  --top-p 0.95
  --read-timeout 7200
  --indices "$INDICES"
  --kvmem-message-expansion
  "${judge_args[@]}"
)

SERVER_CMD_JSON=$(
  printf '%s\0' "${server_cmd[@]}" |
    "$ROOT/.venv/bin/python" -c \
      'import json,sys; print(json.dumps([x.decode() for x in sys.stdin.buffer.read().split(b"\0")[:-1]]))'
)
EVAL_CMD_JSON=$(
  printf '%s\0' "${eval_cmd[@]}" |
    "$ROOT/.venv/bin/python" -c \
      'import json,sys; print(json.dumps([x.decode() for x in sys.stdin.buffer.read().split(b"\0")[:-1]]))'
)
"$ROOT/.venv/bin/python" - "$MANIFEST" "$TAG" "$INDICES" \
    "$SERVER_CMD_JSON" "$EVAL_CMD_JSON" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone

path, tag, indices, server_json, eval_json = sys.argv[1:]
try:
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    git_sha = None
json.dump(
    {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "git_sha": git_sha,
        "indices": indices,
        "controlled_reference": {
            "tag": "longmemeval_m_k224k_query_replay_immutable10_20260722_101213",
            "overall": "6/10",
            "selected_hard4": "0/4",
        },
        "server_command": json.loads(server_json),
        "eval_command": json.loads(eval_json),
        "semantic_groups": "original flattened-history turns; prompt text unchanged",
    },
    open(path, "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2,
)
PY

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

env \
  -u QW3_KVMEM_REBUILT_STATE_DIR \
  QW3_KVMEM_RECOMPUTE_QUERY=1 \
  QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_KVMEM_DUMP_SCORES="$RESULTS/${TAG}_retrieval_scores.jsonl" \
  QW3_Q8_BF16_MAIN=0 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "${server_cmd[@]}" >"$SERVER_LOG" 2>&1 &
server_pid=$!

healthy=0
for _ in $(seq 1 600); do
  if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2 || true
    exit 4
  fi
  sleep 1
done
if [[ "$healthy" -ne 1 ]]; then
  echo "server health timeout; see $SERVER_LOG" >&2
  exit 4
fi

for expected in \
  'kvmem_semantic_expansion=message' \
  'kvmem_group_score_reduce=length-normalized-mass' \
  'kvmem_group_length_alpha=0.5' \
  'kvmem_recompute_query=1' \
  'kvmem_immutable_k=1'; do
  if ! grep -q "$expected" "$SERVER_LOG"; then
    echo "server did not confirm required setting: $expected" >&2
    exit 5
  fi
done

QW3_EVAL_STORE_REASONING=1 \
DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro} \
  "${eval_cmd[@]}" 2>&1 | tee "$RUN_LOG"
