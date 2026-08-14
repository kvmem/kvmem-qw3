#!/usr/bin/env bash
# Queue two resume-safe utility runs behind the current formal latency pass:
#   1. MemoryAgentBench >256K, raw-question score span, K64, B32.
#   2. AgentLongBench 1M canonical first 50, K100 (102400), B32.
#
# This is intentionally a thin supervisor around the existing experiment
# wrappers.  It does not duplicate model/server configuration.
set -Eeuo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
QUEUE_STAMP=${QUEUE_STAMP:-$(date +%Y%m%d_%H%M%S)}
SKIP_LATENCY_WAIT=${SKIP_LATENCY_WAIT:-0}
LATENCY_PID=${LATENCY_PID:-0}
LATENCY_STARTTIME=${LATENCY_STARTTIME:-0}
LATENCY_OUT=${LATENCY_OUT:-/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812}
LATENCY_VALIDATION=${LATENCY_VALIDATION:-$LATENCY_OUT/final_all_validation.json}

MAB_OUT=${MAB_OUT:-/data/chaidi/kvmem_eval/results/memoryagentbench_over256k_raw_query_k64_b32_semantic_chunk_${QUEUE_STAMP}}
MAB_MANIFEST=${MAB_MANIFEST:-$ROOT/scripts/kvmem_eval/memoryagentbench_over256k_contexts.jsonl}
MAB_MAX_ATTEMPTS=${MAB_MAX_ATTEMPTS:-5}

ALB_TAG=${ALB_TAG:-agentlongbench_1m_k100_g32_semantic_chunk2048_fullquery_recent0_adaptive_b32_fp8_mtp4_full50_${QUEUE_STAMP}}
ALB_OUT=${ALB_OUT:-/data/chaidi/kvmem_eval/results/$ALB_TAG}
ALB_MAX_ATTEMPTS=${ALB_MAX_ATTEMPTS:-10}
ALB_PORT=${ALB_PORT:-18118}

LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
STATUS=${STATUS:-$LOG_ROOT/queue_mab_raw_k64_b32_then_alb1m_k100_b32_${QUEUE_STAMP}.status.json}
JUDGE_PY=${JUDGE_PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
phase=initializing

mkdir -p "$LOG_ROOT"

write_status() {
  local state=$1
  local detail=${2:-}
  python3 - "$STATUS" "$state" "$phase" "$detail" "$MAB_OUT" "$ALB_OUT" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2],
    "phase": sys.argv[3],
    "detail": sys.argv[4],
    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "supervisor_pid": os.getppid(),
    "mab_out": sys.argv[5],
    "alb_out": sys.argv[6],
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
PY
}

on_error() {
  local rc=$?
  trap - ERR
  write_status failed "exit=$rc"
  echo "[$(date --iso-8601=seconds)] FAILED phase=$phase exit=$rc" >&2
  exit "$rc"
}
trap on_error ERR

latency_process_is_same() {
  [[ -r /proc/$LATENCY_PID/stat && -r /proc/$LATENCY_PID/cmdline ]] || return 1
  local current_start command
  current_start=$(awk '{print $22}' "/proc/$LATENCY_PID/stat")
  [[ $current_start == "$LATENCY_STARTTIME" ]] || return 1
  command=$(tr '\0' ' ' <"/proc/$LATENCY_PID/cmdline")
  [[ $command == *supervise_paper_latency_qw3_all.sh* ]]
}

latency_artifact_is_valid() {
  python3 - "$LATENCY_VALIDATION" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if report.get("status") != "complete":
    raise SystemExit(1)
if "baseline" not in report or "kvmem" not in report:
    raise SystemExit(1)
PY
}

if [[ $SKIP_LATENCY_WAIT == 1 ]]; then
  echo "[$(date --iso-8601=seconds)] latency wait explicitly skipped"
else
  phase=waiting_latency
  write_status queued "waiting for latency supervisor pid=$LATENCY_PID"
  echo "[$(date --iso-8601=seconds)] waiting for latency pid=$LATENCY_PID starttime=$LATENCY_STARTTIME"
  latency_ended_reported=0
  until latency_artifact_is_valid; do
    if ! latency_process_is_same && (( latency_ended_reported == 0 )); then
      echo "[$(date --iso-8601=seconds)] original latency supervisor ended without a valid final artifact; waiting for a resume/retry"
      write_status queued "original latency run ended; waiting for valid final artifact from retry"
      latency_ended_reported=1
    fi
    sleep 30
  done

  phase=validating_latency
  write_status running "valid latency final artifact detected"
  echo "[$(date --iso-8601=seconds)] validated latency artifact: $LATENCY_VALIDATION"
fi

mab_complete() {
  python3 - "$MAB_OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
progress = root / "progress_summary.json"
if not progress.is_file():
    raise SystemExit(1)
summary = json.loads(progress.read_text(encoding="utf-8"))
if (summary.get("contexts_completed"), summary.get("questions_completed")) != (30, 1316):
    raise SystemExit(1)
records = []
row_summaries = list((root / "rows").glob("*/row_summary.json"))
for path in (root / "rows").glob("*/results.jsonl"):
    records.extend(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
identities = {
    (row.get("split"), row.get("source"), row.get("dataset_row"), row.get("question_index"))
    for row in records
}
if len(row_summaries) != 30 or len(records) != 1316 or len(identities) != 1316:
    raise SystemExit(1)
if any(row.get("query_score_span_mode") != "raw" for row in records):
    raise SystemExit("MAB contains a non-raw query score span")
if any(row.get("score_span_text") != row.get("raw_question") for row in records):
    raise SystemExit("MAB raw score span text does not round-trip")
if any(
    "finish_reason" in row and row.get("finish_reason") not in {"stop", "length"}
    for row in records
):
    raise SystemExit("MAB contains an incomplete generation")
print("validated MAB: contexts=30 questions=1316 query_score_span=raw")
PY
}

phase=running_mab
write_status running "MAB raw-query K64/B32"
for ((attempt = 1; attempt <= MAB_MAX_ATTEMPTS; ++attempt)); do
  echo "[$(date --iso-8601=seconds)] MAB attempt $attempt/$MAB_MAX_ATTEMPTS"
  if OUT_DIR="$MAB_OUT" \
      SELECTION_MANIFEST="$MAB_MANIFEST" \
      BLOCK_TOKENS=32 QUERY_SCORE_SPAN=raw \
      "$ROOT/scripts/kvmem_eval/run_memoryagentbench_over256k_k64_semantic_chunk.sh"; then
    if mab_complete; then
      break
    fi
  fi
  if (( attempt == MAB_MAX_ATTEMPTS )); then
    echo "MAB did not complete after $MAB_MAX_ATTEMPTS attempts" >&2
    false
  fi
  echo "[$(date --iso-8601=seconds)] MAB incomplete; retrying resume-safe output in 30s"
  sleep 30
done

phase=judging_mab
write_status running "MAB generation/local scoring complete; special judge if credential is available"
if [[ -n ${DEEPSEEK_API_KEY:-} ]]; then
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    "$JUDGE_PY" "$ROOT/scripts/kvmem_eval/judge_memoryagentbench_special.py" \
      --results-dir "$MAB_OUT"
  python3 - "$MAB_OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
special = json.loads((root / "special_judge_summary.json").read_text())
final = json.loads((root / "final_summary.json").read_text())
if (special.get("completed"), special.get("expected")) != (316, 316):
    raise SystemExit("incomplete MAB special judge")
if final.get("questions") != 1316:
    raise SystemExit("incomplete MAB final summary")
print(f"validated MAB final score: {100 * final['overall_macro_category_mean']:.4f}%")
PY
else
  echo "[$(date --iso-8601=seconds)] DEEPSEEK_API_KEY is unset; MAB special judge deferred"
fi

alb_complete() {
  python3 - "$ALB_OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
progress = root / "progress.json"
summary = root / "accuracy_summary.json"
config = root / "run_config.json"
if not all(path.is_file() for path in (progress, summary, config)):
    raise SystemExit(1)
p = json.loads(progress.read_text(encoding="utf-8"))
c = json.loads(config.read_text(encoding="utf-8"))
if (p.get("total"), p.get("answers"), p.get("evaluated")) != (50, 50, 50):
    raise SystemExit(1)
if c.get("selected_samples") != 50:
    raise SystemExit("ALB selection is not canonical first-50")
for name in ("answers.jsonl", "eval.jsonl", "manifest.jsonl"):
    rows = [line for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 50:
        raise SystemExit(f"{name} has {len(rows)} rows instead of 50")
print("validated ALB-1M: canonical_first50 answers=50 evaluated=50")
PY
}

phase=running_alb1m
write_status running "ALB-1M canonical first-50 K100/B32"
for ((attempt = 1; attempt <= ALB_MAX_ATTEMPTS; ++attempt)); do
  echo "[$(date --iso-8601=seconds)] ALB-1M attempt $attempt/$ALB_MAX_ATTEMPTS"
  if PORT="$ALB_PORT" KVMEM_BUDGET=102400 BLOCK_TOKENS=32 \
      TAG="$ALB_TAG" \
      METHOD=kvmem_adaptive_k100_g32_b32_semantic_chunk2048_fullquery_recent0_fp8_mtp4 \
      LIMIT=50 START_INDEX=0 SAMPLE_ORDER=dataset \
      "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_k64_semantic_chunk_full50.sh"; then
    if alb_complete; then
      break
    fi
  fi
  if (( attempt == ALB_MAX_ATTEMPTS )); then
    echo "ALB-1M did not complete after $ALB_MAX_ATTEMPTS attempts" >&2
    false
  fi
  echo "[$(date --iso-8601=seconds)] ALB-1M incomplete; retrying resume-safe output in 30s"
  sleep 30
done

phase=complete
write_status complete "latency, MAB, and ALB-1M validated"
echo "[$(date --iso-8601=seconds)] queue complete"
echo "MAB_OUT=$MAB_OUT"
echo "ALB_OUT=$ALB_OUT"
