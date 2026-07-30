#!/usr/bin/env bash
# Controlled AgentLongBench-1M probe for the Fixed-4 Key-direction prototype
# scorer.  By default it uses the ten existing cross-task IDs from the
# round-padding A/B.  Callers may explicitly export an empty QUESTION_IDS and
# set LIMIT/EXPECTED to run a larger prefix of the canonical 50-row subset.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

DEFAULT_QUESTION_IDS="2167e18164c2bfeffc68feb4ebc9e7a66204e04c167289a9643cfcebc267f021,\
3b3f472b43854d64e5dc4a984dea0f993bcefe08630d0295dcc844c14dfbef76,\
3c7151b28bf4845c9c4480ed8782a5085550eac3dc840e0d22e0626f3c15d6f6,\
8176925a596b177f4e8655785ddd744e3bb3e886d512bddc892c223b9548536c,\
daea520e8f6883a738e966bde243df4ab0434d5482eddd529cc359d053b1be40,\
cad128acfc7c410134bcd250e4036a9d6a337abfb4a6674a6a086f55e7ae71cc,\
50e4dad103f3876d3be4b1ab49e648529c49ae3999b70cd682893793ba1b434b,\
e9f1123e4c0a0e12ee6c6219b972463d3742faf497400ad3ede728ca9a7beaac,\
3dd137b4815f91d942915dc978de380ec4fb7d48685c1e93bd9f18f8f4b0b824,\
460a3ec48d3007d80f68bf8a81b973d806a1350a43b205a2191b4c6c4081117b"
QUESTION_IDS=${QUESTION_IDS-$DEFAULT_QUESTION_IDS}

export PORT=${PORT:-8088}
export CTX=1310720
export CPU_GB=${CPU_GB:-80}
export NVME_GB=0
export KVMEM_BUDGET=229376
export GEN_BUDGET=32768
export KVMEM_BLOCK_TOKENS=${KVMEM_BLOCK_TOKENS:-32}
export KVMEM_RETRIEVAL_METHOD=key-direction-fixed4
export KV_DTYPE=${KV_DTYPE:-fp8}
export PREFILL_CHUNK=2048
export GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-0.5}
export TEMP=0.6
export THINKING_BUDGET=8192
export DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl
export MANIFEST=/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl
export BENCHMARK_NAME=${BENCHMARK_NAME:-AgentLongBench-1M-keydir-fixed4-10}
export TAG=${TAG:-agentlongbench_1m_k224k_g32k_b32_keydir_fixed4_max_10_${STAMP}}
export METHOD=${METHOD:-kvmem_keydir_fixed4_max_k224k_g32k_b32_query_replay_immutable_mtp4_kvfp8_idxfp16_qfp16_t06_think8k}
export RESULT_ROOT=${RESULT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
export QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1
export QW3_KVMEM_DUMP_SCORES=${QW3_KVMEM_DUMP_SCORES:-$RESULT_ROOT/retrieval_scores.jsonl}
export QUESTION_IDS
export EXPECTED=${EXPECTED:-10}

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh"
