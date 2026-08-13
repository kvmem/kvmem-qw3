#!/usr/bin/env bash
set -euo pipefail

QW3_ROOT="/home/chaidi/qw3"
QW3_BIN="${QW3_ROOT}/build/qw3"
QW3_MODEL="${QW3_ROOT}/models/Qwen3.6-27B-Q8_0.gguf"
QW3_HISTORY="${QW3_ROOT}/results/kvmem_archive_beam10m_c1/history.qwen-chat.txt"
QW3_RESULTS_ROOT=${QW3_RESULTS_ROOT:-"${QW3_ROOT}/results/workspace_scalability_64k"}
QW3_NVME_ROOT=${QW3_NVME_ROOT:-"/home/chaidi/kvmem_workspace_scalability_nvme"}

MODE="formal"
ADAPTIVE_SCORE_MODE=${ADAPTIVE_SCORE_MODE:-auto}
BLOCK_TOKENS=${BLOCK_TOKENS:-64}
RETRIEVAL_METHOD=${RETRIEVAL_METHOD:-key-direction-adaptive}
EXPECTED_INDEX=${EXPECTED_INDEX:-adaptive}
MIN_FREE_GIB=${MIN_FREE_GIB:-450}
# Optional aggregate Host-State budget.  The storage engine's --cpu-gb flag
# budgets only the CPU-resident KV tier, while the paper table reports that KV
# plus the host-resident retrieval index.  Supplying both values keeps the
# reported Host State under one explicit aggregate limit.
HOST_STATE_GIB=${HOST_STATE_GIB:-}
HOST_INDEX_GIB=${HOST_INDEX_GIB:-}
if [[ "${1:-}" == "--smoke" ]]; then
    MODE="smoke"
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--smoke]" >&2
    exit 2
fi

for required in "${QW3_BIN}" "${QW3_MODEL}" "${QW3_HISTORY}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required file: ${required}" >&2
        exit 1
    fi
done

mapfile -t ACTIVE_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#ACTIVE_GPU_PIDS[@]} != 0 )); then
    echo "GPU is busy; refusing to contaminate the measurement." >&2
    printf 'active compute PID: %s\n' "${ACTIVE_GPU_PIDS[@]}" >&2
    exit 75
fi

AVAILABLE_BYTES=$(df --output=avail -B1 /home/chaidi | tail -n 1 | tr -d ' ')
if [[ "${MODE}" == "formal" ]]; then
    MIN_FREE_BYTES=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
    if (( AVAILABLE_BYTES < MIN_FREE_BYTES )); then
        echo "formal run requires at least ${MIN_FREE_GIB} GiB free on /home/chaidi" >&2
        exit 1
    fi
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${MODE}"
RESULT_DIR="${QW3_RESULTS_ROOT}/${RUN_TAG}"
NVME_DIR="${QW3_NVME_ROOT}/${RUN_TAG}"
mkdir -p "${RESULT_DIR}" "${NVME_DIR}"

if [[ "${MODE}" == "formal" ]]; then
    LADDER="256K,512K,1M,2M,4M,9998336"
    PREFILL_REPEATS=32
    QUERY_REPEATS=17
    GPU_RATIO=0.5
    CPU_GB=32
    if [[ -n "${HOST_STATE_GIB}" ]]; then
        if [[ -z "${HOST_INDEX_GIB}" ]]; then
            echo "HOST_INDEX_GIB is required with HOST_STATE_GIB" >&2
            exit 2
        fi
        CPU_GB=$(python3 -c \
            'import sys; total=float(sys.argv[1]); index=float(sys.argv[2]); assert total > index >= 0; print(f"{total-index:.9f}")' \
            "${HOST_STATE_GIB}" "${HOST_INDEX_GIB}")
    fi
    NVME_GB=384
    TIMEOUT_S=43200
else
    # This result is a parser/tier-path validation artifact, not a paper row.
    LADDER="64K,128K"
    PREFILL_REPEATS=3
    QUERY_REPEATS=2
    # The fixed 64K selection window plus 32K generation reserve needs about
    # 3.5 GiB of resident FP8 KV in addition to model weights and scratch.
    GPU_RATIO=0.38
    CPU_GB=4
    NVME_GB=8
    TIMEOUT_S=1800
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

{
    date --iso-8601=seconds
    git -C "${QW3_ROOT}" rev-parse HEAD
    git -C "${QW3_ROOT}" status --short
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
    lscpu
    lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,FSTYPE,MOUNTPOINTS
    df -h /home/chaidi
    printf 'host_state_budget_gib=%s\n' "${HOST_STATE_GIB:-unbounded}"
    printf 'host_index_reserved_gib=%s\n' "${HOST_INDEX_GIB:-unspecified}"
    printf 'cpu_kv_budget_gib=%s\n' "${CPU_GB}"
} >"${RESULT_DIR}/platform.txt"

set +e
python3 "${QW3_ROOT}/scripts/kvmem_session_profile.py" \
    --qw3 "${QW3_BIN}" \
    --model "${QW3_MODEL}" \
    --input "${QW3_HISTORY}" \
    --ladder "${LADDER}" \
    --decode-tokens 256 \
    --query-tokens 128 \
    --repeat-queries "${QUERY_REPEATS}" \
    --repeat-mode frozen \
    --prefill-probe-tokens 2048 \
    --prefill-probe-repeats "${PREFILL_REPEATS}" \
    --temp 0 \
    --chain 4 \
    --window 65536 \
    --gen-budget 32768 \
    --block-tokens "${BLOCK_TOKENS}" \
    --sink-tokens 512 \
    --recent-tokens 0 \
    --method retrieval \
    --retrieval-method "${RETRIEVAL_METHOD}" \
    --adaptive-gain-1to2 0.10 \
    --adaptive-gain-2to4 0.06 \
    --adaptive-score-mode "${ADAPTIVE_SCORE_MODE}" \
    --index-placement cpu \
    --index-staging-mb 64 \
    --numa-policy auto \
    --gpu-ratio "${GPU_RATIO}" \
    --cpu-gb "${CPU_GB}" \
    --nvme-gb "${NVME_GB}" \
    --nvme-dir "${NVME_DIR}" \
    --kv-dtype fp8 \
    --prefill-chunk 2048 \
    --opt-stage-out on \
    --opt-stage-in on \
    --opt-pack on \
    --raw-k-nvme \
    --tier-threshold 1 \
    --no-deterministic \
    --timeout "${TIMEOUT_S}" \
    --out-json "${RESULT_DIR}/profile.json" \
    2>&1 | tee "${RESULT_DIR}/runner.log"
PROFILE_RC=${PIPESTATUS[0]}
set -e

if [[ -f "${RESULT_DIR}/profile.json" ]]; then
    INTERNAL_LOG=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["log_path"])' \
        "${RESULT_DIR}/profile.json")
    if [[ -f "${INTERNAL_LOG}" ]]; then
        cp -- "${INTERNAL_LOG}" "${RESULT_DIR}/binary.log"
    fi
fi
if (( PROFILE_RC != 0 )); then
    echo "profile failed with exit code ${PROFILE_RC}" >&2
    echo "result_dir=${RESULT_DIR}" >&2
    exit "${PROFILE_RC}"
fi

if [[ "${MODE}" == "formal" ]]; then
    python3 "${QW3_ROOT}/scripts/kvmem_eval/summarize_workspace_scalability.py" \
        "${RESULT_DIR}/profile.json" \
        --expected-index "${EXPECTED_INDEX}" \
        --output "${RESULT_DIR}/table.json" |
        tee "${RESULT_DIR}/table.md"
fi

echo "result_dir=${RESULT_DIR}"
echo "nvme_dir=${NVME_DIR}"
