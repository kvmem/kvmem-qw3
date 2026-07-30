#!/usr/bin/env bash
# Wait for an existing same-parameter probe, seed all completed/evaluated rows
# into a clean full-50 result root, then run only the remaining canonical rows.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to the existing result directory}
SOURCE_PID=${SOURCE_PID:-}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
TAG=${TAG:-agentlongbench_1m_k224k_g32k_b512_s32_keydir_fixed4_max_full50_${STAMP}}
RESULT_ROOT=${RESULT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl
BENCHMARK_NAME=AgentLongBench-1M-keydir-fixed4-B512-S32-50
METHOD=kvmem_keydir_fixed4_max_k224k_g32k_b512_s32_query_replay_immutable_mtp4_kvfp8_idxfp16_qfp16_t06_think8k

if [[ -n "$SOURCE_PID" ]]; then
  echo "[continuation] waiting for same-parameter probe pid=$SOURCE_PID"
  while kill -0 "$SOURCE_PID" 2>/dev/null; do
    sleep 30
  done
fi

"$ROOT/.venv/bin/python" - \
  "$SOURCE_ROOT" "$RESULT_ROOT" "$DATA" "$BENCHMARK_NAME" "$METHOD" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
dataset_path = Path(sys.argv[3])
benchmark = sys.argv[4]
method = sys.argv[5]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_by_id(rows: list[dict]) -> dict[str, dict]:
    return {
        str(row["stable_sample_id"]): row
        for row in rows
        if row.get("stable_sample_id")
    }


samples = read_jsonl(dataset_path)
positions = {
    str(row["stable_sample_id"]): index
    for index, row in enumerate(samples, start=1)
}
answers = latest_by_id(read_jsonl(source / "answers.jsonl"))
evals = latest_by_id(read_jsonl(source / "eval.jsonl"))
manifests = latest_by_id(read_jsonl(source / "manifest.jsonl"))
completed = [sid for sid in positions if sid in evals and sid in answers]

target.mkdir(parents=True, exist_ok=True)
for filename, rows in (
    ("answers.jsonl", answers),
    ("eval.jsonl", evals),
    ("manifest.jsonl", manifests),
):
    with (target / filename).open("w", encoding="utf-8") as handle:
        for sid in completed:
            row = rows.get(sid)
            if row is None:
                continue
            row = dict(row)
            row["benchmark"] = benchmark
            row["method"] = method
            row["index"] = positions[sid]
            row["total"] = len(samples)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

score_dump = source / "retrieval_scores.jsonl"
if score_dump.exists():
    shutil.copyfile(score_dump, target / "retrieval_scores.jsonl")

print(
    f"[continuation] seeded {len(completed)}/{len(samples)} completed rows "
    f"from {source} into {target}",
    flush=True,
)
PY

export TAG RESULT_ROOT BENCHMARK_NAME METHOD
exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_keydir_fixed4_b512_50.sh"
