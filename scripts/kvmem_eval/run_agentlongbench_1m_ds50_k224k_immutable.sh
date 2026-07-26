#!/usr/bin/env bash
# AgentLongBench-Long fixed 1M DeepSeek-comparison 50 using the shared evaluator.
# This is only a parameter wrapper: prompt construction and official evaluation
# remain in run_agentlongbench_kvmem.py via the existing unified launcher.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

# 1.25M logical capacity covers the official 1M bucket plus the 32K decode
# reserve. Historical immutable raw-K and V remain CPU-only; no NVMe tier is
# enabled for this experiment.
export PORT=${PORT:-8088}
export CTX=${CTX:-1310720}
export CPU_GB=${CPU_GB:-80}
export NVME_GB=${NVME_GB:-0}
export KVMEM_BUDGET=${KVMEM_BUDGET:-229376}
export GEN_BUDGET=${GEN_BUDGET:-32768}
export KV_DTYPE=${KV_DTYPE:-fp8}
export PREFILL_CHUNK=${PREFILL_CHUNK:-2048}

# Qwen3 thinking-mode recommended sampling recipe. The shared launcher sends
# top_p=0.95; qw3's explicit defaults retain top_k=20, min_p=0, presence
# penalty=0, and repetition penalty=1.
export TEMP=${TEMP:-0.6}
export THINKING_BUDGET=${THINKING_BUDGET:-8192}

export DATA=${DATA:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
export MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl}
export BENCHMARK_NAME=${BENCHMARK_NAME:-AgentLongBench-1M-DS50}
export TAG=${TAG:-agentlongbench_1m_ds50_k224k_g32k_b32_meank_kvfp8_idxqfp16_t06_think8k_cpu_all_on_${STAMP}}
export METHOD=${METHOD:-kvmem_mean_k_k224k_g32k_b32_query_replay_immutable_mtp4_kvfp8_idxfp16_qfp16_t06_think8k_cpu_all_on}
export LIMIT=${LIMIT:-50}
export EXPECTED=${EXPECTED:-50}

# Empty selectors mean the paper-facing performance optimizations are all on.
export KVMEM_OPT_LEVEL=${KVMEM_OPT_LEVEL:-}
export KVMEM_OPTIMIZE_OFF=${KVMEM_OPTIMIZE_OFF:-}

# The DeepSeek CompactOnly, Compact+RAG, and sliding-window baselines all use
# the same ordered 50 IDs. They are exactly the first 50 IDs of the frozen
# DeepseekMillion parent manifest. Refuse to run if that relationship changes.
"$ROOT/.venv/bin/python" - \
  "$MANIFEST" \
  /home/chaidi/AgentLongBench-Long/results/deepseek_million_qwen_q8_compact_only_50/manifests/compact_only.manifest.jsonl \
  "$LIMIT" <<'PY'
import json
import sys

parent_path, reference_path, limit_text = sys.argv[1:]
limit = int(limit_text)

def ids(path):
    with open(path, encoding="utf-8") as handle:
        return [
            str(json.loads(line)["stable_sample_id"])
            for line in handle
            if line.strip()
        ]

parent = ids(parent_path)[:limit]
reference = ids(reference_path)
if parent != reference:
    raise SystemExit(
        "AgentLongBench 1M DS50 ID mismatch: the selected parent prefix "
        "does not match the DeepSeek comparison manifest"
    )
print(f"AgentLongBench 1M DS50 ID validation passed: {len(reference)} ordered IDs")
PY

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh"
