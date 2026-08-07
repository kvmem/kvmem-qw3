# Chunked reselection efficiency optimization (2026-08-07)

## Scope

This optimization keeps the K64 semantic-chunk experiment unchanged:

- 2,048-token prefill chunks and full-chunk retrieval queries;
- 64K active KVMem budget plus 32K generation reserve;
- 32-token physical blocks, GPU-resident FP16 Adaptive index, FP8 KV;
- exact Adaptive prototypes, softmax, MaxSim, budget, and Top-K semantics;
- immutable K, query replay, MTP-4, and all stage/pack optimizations enabled.

The performance request asks for one output token while retaining the normal
32K server-side generation reserve. It is therefore a TTFT/perfill benchmark,
not a utility result.

## Bottleneck and implementation

The former exact scorer bounded its `Q * heads * blocks` statistics workspace
by splitting a 2,048-token query into as many as 12 query tiles. Every query
tile rescanned and repacked the complete Adaptive index. At 512K history this
made retrieval scoring the largest avoidable KVMem cost.

The new GPU-resident path is an exact Tensor-GEMM two-pass scorer:

1. Stage one complete query layer.
2. Scan prototype tiles once to compute the global per-query/head online LSE.
3. Scan the same tiles a second time and directly accumulate normalized block
   MaxSim scores.

Its persistent statistics workspace is `O(Q * heads)`, while the bounded
logits tile continues to use `QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB` (512 MiB in
the experiment). Thus every prototype dot product is evaluated twice, but the
index is scanned exactly twice instead of once per query tile. No scorer VRAM
ceiling was increased.

Additional changes:

- Semantic-chunk rollback stashes the host query in place, avoiding two
  384-MiB host copies per event.
- Queries up to `QW3_KVMEM_QUERY_PINNED_MIB` (default 512 MiB) may be captured
  directly into bounded pinned host storage. Allocation failure safely falls
  back to the existing pageable/bounce path.
- Query-layer H2D is pipelined with scoring of the previous layer and fenced
  before the shared ingress buffer is reused.
- `QW3_KVMEM_ADAPTIVE_GPU_TWO_PASS=0` retains an explicit A/B fallback to the
  old query-chunked scorer. `QW3_KVMEM_ADAPTIVE_TWO_PASS_LAYOUT=natural` is an
  experimental layout; head-major remains the faster default.

## Same-sample A/B result

Both runs use stable sample ID
`2e754542b66666a0d6d86ee81f8ac34c1d6e2c66f128a95efc8ee6d1c73460d8`
and execute 221 KVMem retrieval operations.

| Metric | Old exact query-chunked scorer | Optimized exact two-pass | Change |
|---|---:|---:|---:|
| Engine / pre-answer time | 515.396 s | 459.914 s | **-10.76%** |
| Retrieval scoring total | 143.116 s | 89.443 s | **-37.50%** |
| Retrieval scoring / operation | 647.584 ms | 404.717 ms | **-37.50%** |
| Exposed reselection total | 173.979 s | 118.376 s | **-31.96%** |
| Exposed reselection / operation | 787.234 ms | 535.638 ms | **-31.96%** |
| Stage-in total | 13.906 s | 13.318 s | timing variance |
| Stage-out total | 0.658 s | 0.732 s | timing variance |
| Assembly / re-RoPE total | 25.573 s | 23.297 s | timing variance |

The stage sub-times overlap, so they must not be summed and compared directly
with exposed reselection time.

Validation:

- CUDA score parity: max absolute error `2.235e-08`, max relative error
  `2.257e-07` against the scalar exact implementation.
- Unit-test Top-K intersection: `8/8`.
- All 221 logged Top-6 block IDs match the old scorer, and all 221 selection
  reuse/transition records are byte-identical.
- No scorer fallback occurred.
- Relevant KVMem store, assembly, immutable-K, scorer, and Mean-K merge tests
  all pass.

Raw logs:

- Old scorer: `/data/chaidi/kvmem_eval/logs/agentlongbench_512k_k64_chunk_legacy_scorer_perf_20260807_server.log`
- Optimized scorer: `/data/chaidi/kvmem_eval/logs/agentlongbench_512k_k64_chunk_twopass_pinned_final_perf_20260807_server.log`

## Remaining irreducible cost under current semantics

The optimized run still spends about 156.2 s in provisional target-model
forward and 164.5 s in replay. Chunked reselection intentionally performs both:
the provisional pass constructs the context-conditioned retrieval query, while
the replay pass builds durable KV/index state under the selected window. Removing
either pass would change the algorithm. Further large gains therefore require
an inline/layer-streamed query-scoring design or a semantic change, not another
local I/O copy optimization.
