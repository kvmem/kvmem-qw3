#!/usr/bin/env bash
# Full efficiency benchmark: llama sweep + CB/MTP matrix (~4-8 h).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_benchmark_common.sh"

load_manifest
init_result_dir efficiency
require_binary "$QW3" "qw3"
require_model

TIMEOUT="${EFFICIENCY_TIMEOUT:-3600}"
: "${LLAMA_COMPLETION:?set LLAMA_COMPLETION in benchmark/manifest.env}"
require_binary "$LLAMA_COMPLETION" "llama-completion"

PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
CTX="${CTX:-262144}"
KV_DTYPE="${KV_DTYPE:-fp16}"
KV_POOL_PAGES="${KV_POOL_PAGES:-49152}"

step "Block A: llama.cpp sweep (default qw3)"
python3 scripts/long_prompt_sweep.py \
  --qw3 "$QW3" \
  --llama "$LLAMA_COMPLETION" \
  --model "$MODEL" \
  --prompt-tokens "4096 8192 16384 32768 65536 131072" \
  --trials 3 \
  -n 512 \
  -c "$CTX" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --sample-peak \
  --variant default \
  --timeout "$TIMEOUT" \
  --json "${OUT}/llama_sweep_default.json" || echo "WARN: llama_sweep_default had failures (continuing)"

step "Block A: llama.cpp sweep (paged KV)"
python3 scripts/long_prompt_sweep.py \
  --qw3 "$QW3" \
  --llama "$LLAMA_COMPLETION" \
  --model "$MODEL" \
  --prompt-tokens "4096 32768 65536 131072" \
  --trials 3 \
  -n 512 \
  -c "$CTX" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --paged-kv \
  --sample-peak \
  --variant paged_kv \
  --timeout "$TIMEOUT" \
  --json "${OUT}/llama_sweep_paged_kv.json" || echo "WARN: llama_sweep_paged_kv had failures (continuing)"

step "Block B: continuous batching matrix"
python3 scripts/continuous_batching_benchmark.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --variants 'plain continuous body recurrent' \
  --ctx-sizes '4096' \
  --input-token-targets '4096' \
  --concurrency-levels '1 2 4 8' \
  --max-tokens 512 \
  --prefill-chunk "$PREFILL_CHUNK" \
  --max-active 8 \
  --timing \
  --ignore-eos \
  --reuse-server \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/cb_matrix_4k.json" || echo "WARN: cb_matrix_4k had failures (continuing)"

python3 scripts/continuous_batching_benchmark.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --variants 'plain continuous body recurrent' \
  --ctx-sizes '32768' \
  --input-token-targets '32768' \
  --concurrency-levels '1 2 4' \
  --max-tokens 512 \
  --prefill-chunk "$PREFILL_CHUNK" \
  --max-active 4 \
  --timing \
  --ignore-eos \
  --reuse-server \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/cb_matrix_32k.json" || echo "WARN: cb_matrix_32k had failures (continuing)"

step "Block C: MTP single-request (CB off, 4K)"
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --modes 'legacy legacy_mtp' \
  --concurrency-levels '1' \
  --ctx 4096 \
  --max-tokens 128 \
  --prompt-repeat 32 \
  --chain 4 \
  --kv-dtype "$KV_DTYPE" \
  --max-active 4 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/mtp_legacy_4k.json" || echo "WARN: mtp_legacy_4k had failures (continuing)"

step "Block C: MTP continuous batching (4K)"
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --modes 'continuous continuous_mtp' \
  --concurrency-levels '1 2 4' \
  --ctx 4096 \
  --max-tokens 128 \
  --prompt-repeat 32 \
  --chain 4 \
  --kv-dtype "$KV_DTYPE" \
  --max-active 4 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/mtp_cb_4k.json" || echo "WARN: mtp_cb_4k had failures (continuing)"

step "Block C: long-context MTP spot (32K, no MTP)"
python3 scripts/long_prompt_sweep.py \
  --qw3 "$QW3" \
  --llama "$LLAMA_COMPLETION" \
  --model "$MODEL" \
  --prompt-tokens "32768" \
  --trials 2 \
  -n 1024 \
  -c "$CTX" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --paged-kv \
  --variant longctx_32k_nomtp \
  --timeout "$TIMEOUT" \
  --json "${OUT}/mtp_longctx_32k_nomtp.json" || echo "WARN: mtp_longctx_32k_nomtp had failures (continuing)"

step "Block C: long-context MTP spot (32K, MTP chain 4)"
python3 scripts/long_prompt_sweep.py \
  --qw3 "$QW3" \
  --llama "$LLAMA_COMPLETION" \
  --model "$MODEL" \
  --prompt-tokens "32768" \
  --trials 2 \
  -n 1024 \
  -c "$CTX" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --paged-kv \
  --mtp-chain 4 \
  --variant longctx_32k_mtp4 \
  --timeout "$TIMEOUT" \
  --json "${OUT}/mtp_longctx_32k_mtp4.json" || echo "WARN: mtp_longctx_32k_mtp4 had failures (continuing)"

step "Block C: long-context MTP spot (64K / 128K, record only)"
for target in 65536 131072; do
  python3 scripts/long_prompt_sweep.py \
    --qw3 "$QW3" \
    --llama "$LLAMA_COMPLETION" \
    --model "$MODEL" \
    --prompt-tokens "$target" \
    --trials 2 \
    -n 1024 \
    -c "$CTX" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --paged-kv \
    --mtp-chain 4 \
    --variant "longctx_${target}_mtp4" \
    --timeout "$TIMEOUT" \
    --json "${OUT}/mtp_longctx_${target}_mtp4.json" || true
done

step "Block D: FP8 KV spot (4K)"
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --modes 'continuous continuous_mtp' \
  --concurrency-levels '4' \
  --ctx 4096 \
  --max-tokens 128 \
  --prompt-repeat 32 \
  --chain 4 \
  --kv-dtype fp8 \
  --max-active 4 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/mtp_cb_4k_fp8.json" || true

step "summarize"
python3 scripts/summarize_benchmark_results.py \
  --kind efficiency \
  --input-dir "$OUT" \
  --output "${OUT}/summary_efficiency.md"

echo "efficiency benchmark finished -> ${OUT}"
