#!/usr/bin/env bash
set -euo pipefail

# Full qw3 vs llama.cpp benchmark requested for Qwen3.6 27B:
#   input lengths: 1K, 2K, 4K, 8K, 16K, 64K, 128K, 250K target
#   output length: 1K
#   modes: plain and MTP
#   metrics: prefill/decode throughput, TTFT, TBT, peak GPU VRAM
#
# Results are checkpointed after every cell. Re-running this script resumes and
# skips successful rows unless FORCE=1 is set.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${OUT_DIR:-/tmp/qw3_llama_full_bench}"
MODEL="${MODEL:-${ROOT}/models/Qwen3.6-27B-Q8_0.gguf}"
QW3="${QW3:-${ROOT}/build/qw3}"
LLAMA_SERVER="${LLAMA_SERVER:-/tmp/llama.cpp/build-cuda/bin/llama-server}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/qw3_llama_full_1kout.json}"
OUT_HTML="${OUT_HTML:-${OUT_DIR}/qw3_llama_full_1kout.html}"
LLAMA_PORT="${LLAMA_PORT:-18099}"
TRIALS="${TRIALS:-3}"
MTP_CHAIN="${MTP_CHAIN:-2 3 4 5}"

mkdir -p "${OUT_DIR}"

extra=()
if [[ "${FORCE:-0}" == "1" ]]; then
  extra+=(--force)
else
  extra+=(--resume)
fi

python3 scripts/bench/run_bench.py \
  --full-1kout \
  "${extra[@]}" \
  --model "${MODEL}" \
  --qw3 "${QW3}" \
  --llama-server "${LLAMA_SERVER}" \
  --llama-port "${LLAMA_PORT}" \
  --trials "${TRIALS}" \
  --mtp-chain "${MTP_CHAIN}" \
  --out "${OUT_JSON}" \
  --html "${OUT_HTML}"

cat <<EOF

Benchmark JSON:
  ${OUT_JSON}

Static HTML report:
  ${OUT_HTML}

Open report:
  ${OUT_HTML}

Optional HTTP server:
  cd ${OUT_DIR}
  python3 -m http.server 8017
  # then open http://127.0.0.1:8017/$(basename "${OUT_HTML}")

Resume later:
  ${BASH_SOURCE[0]}

Force rerun:
  FORCE=1 ${BASH_SOURCE[0]}
EOF
