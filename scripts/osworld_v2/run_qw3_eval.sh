#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"
SETUP_EXPORTS="${OSWORLD_SETUP_EXPORTS:-${OSWORLD_ROOT}/cache/osworld-setup-exports.sh}"
GITLAB_EXPORTS="${OSWORLD_GITLAB_EXPORTS:-${QW3_ROOT}/third_party/OSWorld-GitLab/.env.osworld.local}"

if [[ -f "${SETUP_EXPORTS}" ]]; then
  # This ignored, mode-0600 file is the designated location for local secrets.
  # shellcheck disable=SC1090
  source "${SETUP_EXPORTS}"
fi
if [[ -f "${GITLAB_EXPORTS}" ]]; then
  # shellcheck disable=SC1090
  source "${GITLAB_EXPORTS}"
fi

TEST_META="${OSWORLD_TEST_META:-${OSWORLD_ROOT}/evaluation_examples/test_v2.json}"
RUN_LABEL="${QW3_OSWORLD_RUN_LABEL:-full}"
PORT="${QW3_OSWORLD_PORT:-18080}"
MAX_STEPS="${QW3_OSWORLD_MAX_STEPS:-500}"
MAX_OUTPUT_TOKENS="${QW3_OSWORLD_MAX_OUTPUT_TOKENS:-4096}"
CONTEXT_TOKENS="${QW3_OSWORLD_CONTEXT_TOKENS:-65536}"
SAMPLING_TEMPERATURE="${QW3_OSWORLD_TEMPERATURE:-1.0}"
SAMPLING_TOP_P="${QW3_OSWORLD_TOP_P:-0.95}"
SAMPLING_TOP_K="${QW3_OSWORLD_TOP_K:-20}"
SAMPLING_MIN_P="${QW3_OSWORLD_MIN_P:-0.0}"
SAMPLING_PRESENCE_PENALTY="${QW3_OSWORLD_PRESENCE_PENALTY:-0.0}"
SAMPLING_REPETITION_PENALTY="${QW3_OSWORLD_REPETITION_PENALTY:-1.0}"
THINKING_BUDGET="${QW3_OSWORLD_THINKING_BUDGET:-3072}"
ENABLE_THINKING="${QW3_OSWORLD_ENABLE_THINKING:-1}"
# Action requests explicitly select their thinking mode. Keep the server's
# fallback independently configurable for evaluator/user-simulator requests.
SERVER_ENABLE_THINKING="${QW3_OSWORLD_SERVER_ENABLE_THINKING:-${ENABLE_THINKING}}"
export QW3_OSWORLD_HISTORY_MODE="${QW3_OSWORLD_HISTORY_MODE:-sliding}"
export QW3_OSWORLD_CONTEXT_TOKENS="${CONTEXT_TOKENS}"
export QW3_OSWORLD_SUMMARY_TRIGGER_TOKENS="${QW3_OSWORLD_SUMMARY_TRIGGER_TOKENS:-196608}"
export QW3_OSWORLD_SUMMARY_MAX_TOKENS="${QW3_OSWORLD_SUMMARY_MAX_TOKENS:-4096}"
if [[ "${QW3_OSWORLD_HISTORY_MODE}" = "summary" ]]; then
  export QW3_VISION_GPU_CACHE_MIB="${QW3_VISION_GPU_CACHE_MIB:-4096}"
fi
MTP_CHAIN="${QW3_OSWORLD_MTP_CHAIN:-0}"
KVMEM_PREFIX_CACHE="${QW3_OSWORLD_KVMEM_PREFIX_CACHE:-0}"
PREFIX_CACHE="${QW3_OSWORLD_PREFIX_CACHE:-0}"
MAX_ACTIVE="${QW3_OSWORLD_MAX_ACTIVE:-1}"
PREFIX_CACHE_GUARD_PAGES="${QW3_OSWORLD_PREFIX_CACHE_GUARD_PAGES:-1}"
KVMEM_BLOCK_TOKENS="${QW3_OSWORLD_KVMEM_BLOCK_TOKENS:-128}"
KVMEM_GEN_BUDGET="${QW3_OSWORLD_KVMEM_GEN_BUDGET:-${MAX_OUTPUT_TOKENS}}"
# Keep one block outside selection+generation so immutable raw-K runs through
# the bounded tiered pool instead of a dense full-context allocation.
KVMEM_BUDGET="${QW3_OSWORLD_KVMEM_BUDGET:-$((CONTEXT_TOKENS - KVMEM_GEN_BUDGET - KVMEM_BLOCK_TOKENS))}"
KVMEM_CPU_GB="${QW3_OSWORLD_KVMEM_CPU_GB:-2}"
KVMEM_METHOD="${QW3_OSWORLD_KVMEM_METHOD:-recency}"
KVMEM_QUERY_CONDITIONED="${QW3_OSWORLD_KVMEM_QUERY_CONDITIONED:-0}"
KVMEM_INDEX_PLACEMENT="${QW3_OSWORLD_KVMEM_INDEX_PLACEMENT:-gpu}"
export QW3_MESSAGE_PREFIX_CACHE="${QW3_MESSAGE_PREFIX_CACHE:-${QW3_OSWORLD_MESSAGE_PREFIX_CACHE:-0}}"
export QW3_PREPARED_PREFIX_CACHE="${QW3_PREPARED_PREFIX_CACHE:-1}"
QW3_BINARY="${QW3_OSWORLD_BINARY:-${QW3_ROOT}/build/qw3}"
if [[ -n "${QW3_OSWORLD_EVAL_PROFILE:-}" ]]; then
  EVALUATION_PROFILE="${QW3_OSWORLD_EVAL_PROFILE}"
elif [[ "${MAX_STEPS}" -eq 500 ]]; then
  EVALUATION_PROFILE="osworld-v2-public-pyautogui-500"
else
  EVALUATION_PROFILE="exploratory-pyautogui-${MAX_STEPS}"
fi
RESULT_ROOT="${QW3_OSWORLD_RESULT_ROOT:-${QW3_ROOT}/benchmark/osworld_v2/results}"
RUN_ID="${QW3_OSWORLD_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RESULT_ROOT}/${RUN_LABEL}-${RUN_ID}"
SERVER_LOG="${RUN_DIR}/qw3_server.log"
RUNNER_LOG="${RUN_DIR}/runner.log"
VM_IMAGE="${OSWORLD_ROOT}/docker_vm_data/osworld-v2-ubuntu-x86.qcow2"
MODEL="${QW3_OSWORLD_MODEL:-${QW3_ROOT}/models/Qwen3.8-27B-Q8_0.gguf}"
VISION_MODEL="${QW3_OSWORLD_VISION_MODEL:-${QW3_ROOT}/models/Qwen3.8-27B-BF16}"
RUNTIME_REF="happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9"
export UV_CACHE_DIR="${OSWORLD_ROOT}/cache/uv"

die() {
  echo "$*" >&2
  exit 1
}

if pgrep -f 'run_dense256_pass4.py' >/dev/null || \
    pgrep -f 'ordinary_only_batch.py.*dense256_claude_auto_compact' >/dev/null || \
    [[ -n "$(docker ps --filter label=com.qw3.deepswe.run-owner --quiet 2>/dev/null)" ]]; then
  echo "DeepSWE main/worker/container is still active; refusing to start OSWorld." >&2
  exit 75
fi

[[ "$(git -C "${OSWORLD_ROOT}" describe --tags --exact-match)" = "v2026.08.08" ]] || \
  die "OSWorld checkout is not pinned to v2026.08.08."
[[ -f "${TEST_META}" ]] || die "Missing OSWorld task list: ${TEST_META}"
[[ -s "${VM_IMAGE}" ]] || die "Missing extracted OSWorld VM image: ${VM_IMAGE}"
[[ -s "${MODEL}" ]] || die "Missing QW3 model: ${MODEL}"
[[ -f "${VISION_MODEL}/model.safetensors.index.json" ]] || \
  die "Missing QW3 vision model: ${VISION_MODEL}"
task_count="$(find "${OSWORLD_ROOT}/evaluation_examples/task_class" -maxdepth 1 -name 'task_*.py' | wc -l)"
[[ "${task_count}" -eq 108 ]] || \
  die "Expected 108 gated OSWorld task classes, found ${task_count}."
[[ -n "$(find "${OSWORLD_ROOT}/cache/osworld_v2_assets" -mindepth 1 -maxdepth 2 -print -quit)" ]] || \
  die "Missing gated OSWorld task assets."
docker image inspect "${RUNTIME_REF}" >/dev/null || \
  die "Missing pinned Docker runtime image ${RUNTIME_REF}."
[[ "$(docker image inspect happysixd/osworld-docker:latest --format '{{.Id}}')" = \
   "$(docker image inspect "${RUNTIME_REF}" --format '{{.Id}}')" ]] || \
  die "happysixd/osworld-docker:latest does not match the pinned runtime digest."
[[ -e /dev/kvm ]] || die "Missing /dev/kvm."
mkdir -p "${RUN_DIR}"

[[ "${MTP_CHAIN}" =~ ^[0-9]+$ ]] || die "QW3_OSWORLD_MTP_CHAIN must be a non-negative integer."
[[ "${MAX_ACTIVE}" =~ ^[1-9][0-9]*$ ]] || die "QW3_OSWORLD_MAX_ACTIVE must be a positive integer."
[[ "${PREFIX_CACHE_GUARD_PAGES}" =~ ^[0-9]+$ ]] || \
  die "QW3_OSWORLD_PREFIX_CACHE_GUARD_PAGES must be a non-negative integer."

case "${PREFIX_CACHE}" in
  0|1)
    ;;
  *)
    die "QW3_OSWORLD_PREFIX_CACHE must be 0 or 1."
    ;;
esac

if [[ "${PREFIX_CACHE}" = "1" && "${KVMEM_PREFIX_CACHE}" = "1" ]]; then
  die "Ordinary and KVMem prefix caches cannot be enabled together."
fi

# A commit hash alone is insufficient when evaluating a dirty worktree. Keep
# the tracked diff plus exact copies of the untracked source files that define
# the multimodal-prefix-cache and OSWorld integration used by this run.
git -C "${QW3_ROOT}" status --short --branch >"${RUN_DIR}/source_status.txt"
git -C "${QW3_ROOT}" diff --binary HEAD -- >"${RUN_DIR}/source_dirty.patch"
git -C "${OSWORLD_ROOT}" diff --binary HEAD -- >"${RUN_DIR}/osworld_source_dirty.patch"
tar -czf "${RUN_DIR}/osworld_source_untracked_snapshot.tar.gz" \
  -C "${OSWORLD_ROOT}" mm_agents/qw3_summary_history.py mm_agents/qw3_message_transport.py
tar -czf "${RUN_DIR}/source_untracked_snapshot.tar.gz" \
  -C "${QW3_ROOT}" \
  src/multimodal_prefix_cache.hpp \
  src/message_prefix_cache.hpp \
  src/prepared_message_cache.hpp \
  src/vision_request_preparation.hpp \
  src/chat_token_cache.hpp \
  src/global_kv_page_pool.hpp \
  src/request_cancellation.hpp \
  scripts/multimodal_prefix_cache_canary.py \
  scripts/prefix_cache_long_multimodal_canary.py \
  tests/prefix_cache_lifetime_test.cpp \
  tests/request_disconnect_test.cpp \
  docs/osworld_256k_prefix_repair.md \
  scripts/osworld_v2

kvmem_args=()
case "${KVMEM_PREFIX_CACHE}" in
  0)
    ;;
  1)
    (( KVMEM_GEN_BUDGET > 0 )) || die "KVMem generation budget must be positive."
    (( KVMEM_BUDGET > 0 )) || die "KVMem selection budget must be positive."
    (( KVMEM_BUDGET + KVMEM_GEN_BUDGET <= CONTEXT_TOKENS )) || \
      die "KVMem selection + generation budgets exceed context capacity."
    (( KVMEM_BUDGET % KVMEM_BLOCK_TOKENS == 0 )) || \
      die "KVMem selection budget must be block aligned."
    export QW3_KVMEM_PREFIX_CACHE_TRACE=1
    kvmem_args+=(
      --kvmem
      --kvmem-prefix-cache
      --kvmem-block-tokens "${KVMEM_BLOCK_TOKENS}"
      --kvmem-budget "${KVMEM_BUDGET}"
      --kvmem-prefill-budget "${KVMEM_BUDGET}"
      --kvmem-gen-budget "${KVMEM_GEN_BUDGET}"
      --kvmem-method "${KVMEM_METHOD}"
      --kvmem-retrieval-method "${QW3_OSWORLD_KVMEM_RETRIEVAL_METHOD:-mean-k}"
      --kvmem-index-placement "${KVMEM_INDEX_PLACEMENT}"
      --kvmem-opt-stage-out "${QW3_OSWORLD_KVMEM_OPT_STAGE_OUT:-on}"
      --kvmem-update-mode "${QW3_OSWORLD_KVMEM_UPDATE_MODE:-interval}"
      --kvmem-guided-reselect "${QW3_OSWORLD_KVMEM_GUIDED_RESELECT:-off}"
      --kvmem-guided-thinking-tokens "${QW3_OSWORLD_KVMEM_GUIDED_THINKING_TOKENS:-0}"
      --kvmem-guided-query-tokens "${QW3_OSWORLD_KVMEM_GUIDED_QUERY_TOKENS:-256}"
      --kvmem-middecode-trigger-tokens "${QW3_OSWORLD_KVMEM_REFRESH_TOKENS:-28672}"
      --kvmem-middecode-max-refreshes "${QW3_OSWORLD_KVMEM_MIDDECODE_MAX_REFRESHES:-2}"
      --kvmem-nvme-gb 0
      --no-kvmem-raw-k-nvme
      --kvmem-gpu-memory-ratio 0.90
      --kvmem-cpu-gb "${KVMEM_CPU_GB}"
    )
    case "${KVMEM_QUERY_CONDITIONED}" in
      0) ;;
      1) kvmem_args+=(--kvmem-query-conditioned) ;;
      *) die "QW3_OSWORLD_KVMEM_QUERY_CONDITIONED must be 0 or 1." ;;
    esac
    if [[ -n "${QW3_OSWORLD_KVMEM_SINK_TOKENS:-}" ]]; then
      kvmem_args+=(--kvmem-sink-tokens "${QW3_OSWORLD_KVMEM_SINK_TOKENS}")
    fi
    if [[ -n "${QW3_OSWORLD_KVMEM_RECENT_TOKENS:-}" ]]; then
      kvmem_args+=(--kvmem-recent-tokens "${QW3_OSWORLD_KVMEM_RECENT_TOKENS}")
    fi
    ;;
  *)
    die "QW3_OSWORLD_KVMEM_PREFIX_CACHE must be 0 or 1."
    ;;
esac

serve_route_args=(--no-continuous-batching)
if [[ "${PREFIX_CACHE}" = "1" ]]; then
  # The lossless ordinary prefix cache lives on the paged continuous-batching
  # route. Keep max-active at one for single-trajectory latency experiments so
  # batching/concurrency does not become an additional variable.
  export QW3_PREFIX_CACHE_TRACE=1
  # Independently rendered chat turns replace the prior open assistant suffix
  # with its completed message.  Publish one page before that unstable edge so
  # the next accumulated request can reuse the long, exact prefix.
  export QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES="${PREFIX_CACHE_GUARD_PAGES}"
  serve_route_args=(--continuous-batching --prefix-cache --max-active "${MAX_ACTIVE}")
fi

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${QW3_ROOT}"
thinking_args=()
case "${SERVER_ENABLE_THINKING}" in
  1) thinking_args=(--enable-thinking) ;;
  0) ;;
  *) die "QW3_OSWORLD_SERVER_ENABLE_THINKING must be 0 or 1." ;;
esac
"${QW3_BINARY}" serve \
  --model "${MODEL}" \
  --vision-model "${VISION_MODEL}" \
  --vision-device cuda \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --ctx "${CONTEXT_TOKENS}" \
  --kv-dtype "${QW3_OSWORLD_KV_DTYPE:-fp16}" \
  --prefill-chunk 2048 \
  --mtp-chain "${MTP_CHAIN}" \
  "${serve_route_args[@]}" \
  --temp "${SAMPLING_TEMPERATURE}" \
  --top-p "${SAMPLING_TOP_P}" \
  --top-k "${SAMPLING_TOP_K}" \
  "${thinking_args[@]}" \
  --thinking-budget "${THINKING_BUDGET}" \
  -n "${MAX_OUTPUT_TOKENS}" \
  "${kvmem_args[@]}" \
  >"${SERVER_LOG}" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" >"${RUN_DIR}/server.pid"

ready=0
for _ in $(seq 1 180); do
  if env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
      curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -100 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 1
done
test "${ready}" -eq 1

export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-local-key}"
export WEBSITE_HOST_SUFFIX="${WEBSITE_HOST_SUFFIX:-site.hku.icu}"
export OSWORLD_BENCHMARK_RELEASE="osworld-v2-2026.08.08"
export OSWORLD_FILE_BASE_URL="${OSWORLD_FILE_BASE_URL:-${OSWORLD_ROOT}/cache/osworld_v2_assets}"
export OSWORLD_CLIENT_PASSWORD="${OSWORLD_CLIENT_PASSWORD:-osworld-public-evaluation}"
export HF_ENDPOINT="https://hf-mirror.com"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

# Model-based evaluators default to the same local multimodal QW3 endpoint.
# Override these variables with a reference evaluator for comparable scoring.
export OSWORLD_EVAL_MODEL_PROVIDER="${OSWORLD_EVAL_MODEL_PROVIDER:-openai_compatible}"
export OSWORLD_EVAL_MODEL_NAME="${OSWORLD_EVAL_MODEL_NAME:-gpt-qw3-evaluator}"
export OSWORLD_EVAL_MODEL_API_KEY="${OSWORLD_EVAL_MODEL_API_KEY:-dummy-local-key}"
export OSWORLD_EVAL_MODEL_BASE_URL="${OSWORLD_EVAL_MODEL_BASE_URL:-${OPENAI_BASE_URL}}"
export OSWORLD_EVAL_MODEL_TEMPERATURE="${OSWORLD_EVAL_MODEL_TEMPERATURE:-0}"
export OSWORLD_EVAL_MODEL_MAX_OUTPUT_TOKENS="${OSWORLD_EVAL_MODEL_MAX_OUTPUT_TOKENS:-512}"

# HITL tasks use the same local endpoint unless a dedicated simulator is set.
export OSWORLD_USER_SIM_PROVIDER="${OSWORLD_USER_SIM_PROVIDER:-openai_compatible}"
export OSWORLD_USER_SIM_MODEL="${OSWORLD_USER_SIM_MODEL:-gpt-qw3-user-simulator}"
export OSWORLD_USER_SIM_API_KEY="${OSWORLD_USER_SIM_API_KEY:-dummy-local-key}"
export OSWORLD_USER_SIM_BASE_URL="${OSWORLD_USER_SIM_BASE_URL:-${OPENAI_BASE_URL}}"
export OSWORLD_USER_SIM_TEMPERATURE="${OSWORLD_USER_SIM_TEMPERATURE:-0}"
export OSWORLD_USER_SIM_MAX_TOKENS="${OSWORLD_USER_SIM_MAX_TOKENS:-256}"

# Use OSWorld's Qwen-VL adapter so 1000x1000 relative model coordinates are
# converted back to the VM's exact 1920x1080 pixel space.
# Long 16K-token answers can exceed the old 130s timeout. Explicit request
# cancellation still bounds abandoned work if a configured deadline expires.
export OSWORLD_OPENAI_TIMEOUT="${OSWORLD_OPENAI_TIMEOUT:-600}"
export QW3_OSWORLD_REQUEST_CANCEL="${QW3_OSWORLD_REQUEST_CANCEL:-1}"
export OSWORLD_AGENT_KIND="${OSWORLD_AGENT_KIND:-qwen35vl}"
export QW3_OSWORLD_HISTORY_N="${QW3_OSWORLD_HISTORY_N:-100}"
export QW3_OSWORLD_IMAGE_MAX="${QW3_OSWORLD_IMAGE_MAX:-20}"
export QW3_OSWORLD_FOLD_SIZE="${QW3_OSWORLD_FOLD_SIZE:-10}"
export QW3_OSWORLD_ENABLE_THINKING="${QW3_OSWORLD_ENABLE_THINKING:-1}"
export QW3_OSWORLD_PRESERVE_THINKING="${QW3_OSWORLD_PRESERVE_THINKING:-1}"
export QW3_OSWORLD_KVMEM_HARNESS_PINS="${QW3_OSWORLD_KVMEM_HARNESS_PINS:-${KVMEM_PREFIX_CACHE}}"
export QW3_OSWORLD_TOP_K="${SAMPLING_TOP_K}"
export QW3_OSWORLD_MIN_P="${SAMPLING_MIN_P}"
export QW3_OSWORLD_PRESENCE_PENALTY="${SAMPLING_PRESENCE_PENALTY}"
export QW3_OSWORLD_REPETITION_PENALTY="${SAMPLING_REPETITION_PENALTY}"
export QW3_OSWORLD_THINKING_BUDGET="${THINKING_BUDGET}"

recording_args=()
if [[ "${QW3_OSWORLD_ENABLE_RECORDING:-0}" = "1" ]]; then
  recording_args+=(--enable_recording)
fi

vnc_args=()
if [[ "${QW3_OSWORLD_ENABLE_VNC:-0}" = "1" ]]; then
  vnc_args+=(--enable_vnc)
fi

guest_trace_args=()
if [[ "${QW3_OSWORLD_TRACE_GUEST:-1}" = "1" ]]; then
  guest_trace_args+=(--trace_guest)
fi

"${OSWORLD_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/write_run_manifest.py" \
  --output "${RUN_DIR}/run_manifest.json" \
  --qw3-root "${QW3_ROOT}" \
  --osworld-root "${OSWORLD_ROOT}" \
  --test-meta "${TEST_META}" \
  --model "${MODEL}" \
  --vision-model "${VISION_MODEL}" \
  --run-label "${RUN_LABEL}" \
  --run-id "${RUN_ID}" \
  --evaluation-profile "${EVALUATION_PROFILE}" \
  --max-steps "${MAX_STEPS}" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
  --context-tokens "${CONTEXT_TOKENS}" \
  --prefix-cache "${PREFIX_CACHE}" \
  --prefix-cache-guard-pages "${PREFIX_CACHE_GUARD_PAGES}" \
  --continuous-batching "${PREFIX_CACHE}" \
  --max-active "${MAX_ACTIVE}" \
  --kvmem-prefix-cache "${KVMEM_PREFIX_CACHE}" \
  --kvmem-block-tokens "${KVMEM_BLOCK_TOKENS}" \
  --kvmem-budget "${KVMEM_BUDGET}" \
  --kvmem-gen-budget "${KVMEM_GEN_BUDGET}" \
  --kvmem-cpu-gb "${KVMEM_CPU_GB}" \
  --kvmem-immutable-k 1 \
  --port "${PORT}"

# The pinned upstream run_multienv imports the AWS manager even for Docker.
# These placeholders satisfy import-time validation and are never used by the
# selected Docker provider.
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_SUBNET_ID="${AWS_SUBNET_ID:-unused-for-docker}"
export AWS_SECURITY_GROUP_ID="${AWS_SECURITY_GROUP_ID:-unused-for-docker}"

expected_tasks="$("${OSWORLD_ROOT}/.venv/bin/python" -c \
  'import json, sys; print(sum(len(v) for v in json.load(open(sys.argv[1])).values()))' \
  "${TEST_META}")"

cd "${OSWORLD_ROOT}"
uv run python scripts/python/run_multienv.py \
  --provider_name docker \
  --path_to_vm "${VM_IMAGE}" \
  --headless \
  --observation_type screenshot \
  --action_space pyautogui \
  --model gpt-qw3 \
  --temperature "${SAMPLING_TEMPERATURE}" \
  --top_p "${SAMPLING_TOP_P}" \
  --max_tokens "${MAX_OUTPUT_TOKENS}" \
  --max_trajectory_length 1 \
  --max_steps "${MAX_STEPS}" \
  --num_envs 1 \
  --eval_version v2 \
  --client_password "${OSWORLD_CLIENT_PASSWORD}" \
  --test_all_meta_path "${TEST_META}" \
  --result_dir "${RUN_DIR}/osworld" \
  "${recording_args[@]}" \
  "${vnc_args[@]}" \
  "${guest_trace_args[@]}" \
  2>&1 | tee "${RUNNER_LOG}"

# Upstream currently logs process-level failures but may still return zero.
# Treat a run as successful only when every selected task produced result.txt.
result_count="$(find "${RUN_DIR}/osworld" -type f -name result.txt | wc -l)"
[[ "${result_count}" -eq "${expected_tasks}" ]] || \
  die "OSWorld produced ${result_count}/${expected_tasks} task results; see ${RUN_DIR}."

echo "OSWorld result directory: ${RUN_DIR}"
