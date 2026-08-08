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

## V2: exact one-dot scoring and provisional pruning

The next implementation removes the two-pass dot recomputation while keeping
the same 512-MiB scorer ceiling. It partitions that existing slab between a
query tile's FP32 GEMM logits and per-block MaxSim values. A fused reducer
consumes every `q * prototype` logit once, computes the global LSE and block
maximum together, and then finalizes the normalized score. Query tiles are
balanced so a nearly-empty tail does not issue a full-layer GEMM.

The provisional target pass also stops immediately after capturing the last
configured normal-attention Q. Its terminal layer projects Q only; K/V,
attention, output projection, FFN, KV append, and MTP-prefix work that cannot
affect retrieval are skipped. Both changes are exact and default to bounded
fallback paths when their preconditions are not met.

| Metric | Exact two-pass | Exact one-dot + provisional prune | Change |
|---|---:|---:|---:|
| Engine / pre-answer time | 459.914 s | 416.161 s | **-9.51%** |
| Retrieval scoring total | 89.443 s | 50.559 s | **-43.47%** |
| Exposed reselection total | 118.376 s | 79.935 s | **-32.48%** |
| Provisional target forward | 156.154 s | 151.724 s | **-2.84%** |
| Durable replay | 164.5 s | 163.910 s | approximately unchanged |

The maximum logged one-dot workspace is 511.97 MiB, below the same 512-MiB
limit. The implementation does not add an index mirror or raise the GPU
workspace cap. CUDA parity against the scalar exact scorer has maximum absolute
error `2.235e-08`, maximum relative error `2.250e-07`, and Top-K intersection
`8/8`.

Raw log:

- `/data/chaidi/kvmem_eval/logs/agentlongbench_512k_k64_chunk_onedot_qonly_final_perf_20260807_server.log`

## V3: online Q consumption and exact replay-prefix reuse

Profiling V2 showed that each 2,048-token event moved 384 MiB of captured Q
from GPU to CPU and then back to GPU. V3 lets the exact one-dot scorer consume
each normal-attention layer's de-RoPE'd FP16 Q while it is still on GPU. It
reuses the one-dot workspace and one existing query-capture slot, so it adds no
device allocation. This changes accounting, not the score: scorer work moves
from the explicit reselection phase into the provisional phase.

The online path also makes the ordinary 48-MiB query staging allocation idle
between scoring and replay. V3 stores the 40-MiB hidden batch at the input of
the first normal-attention layer in that allocation. For this model the first
normal-attention layer is layer 3, so the durable replay can reuse the exact
output and recurrent/conv state of layers 0--2. Those layers cannot observe the
selected normal-attention KV window. Reuse stops at layer 3; extending it past
that boundary would change hidden states and is not valid.

| Metric | V2 one-dot | V3 online + prefix reuse | Change |
|---|---:|---:|---:|
| Engine / pre-answer time | 416.161 s | 411.291 s | **-1.17%** |
| Provisional phase, including online score | 151.724 s | 201.564 s | scorer moved here |
| Explicit reselection phase | 79.935 s | 30.132 s | **-62.30%** |
| Retrieval time charged inside reselection | 50.559 s | 1.091 s | scorer moved online |
| Durable replay | 163.910 s | 159.346 s | **-2.78%** |
| Stage-in | 13.446 s | 13.182 s | timing variance |
| Assembly / re-RoPE | 24.112 s | 23.389 s | timing variance |

V3 performed 219 online scores and 219 prefix reuses. All 221 score traces and
all 222 working-set transition records are byte-identical to V2. The first
generated token is also identical. The observed device-memory footprint stayed
within the same range; structurally V3 replaces two query transfer slots with
one and stores its hidden snapshot in an already allocated tensor.

Raw logs:

- Online score only: `/data/chaidi/kvmem_eval/logs/agentlongbench_512k_k64_chunk_online_score_same_perf_20260807_server.log`
- Online score plus prefix reuse: `/data/chaidi/kvmem_eval/logs/agentlongbench_512k_k64_chunk_prefixreuse_debug2_20260807_server.log`

Across the complete sequence, engine time falls from 515.396 s to 411.291 s,
a **20.20%** cumulative reduction without changing retrieval or increasing the
configured GPU-memory ceiling.

## 32K utility follow-up: minimal recent anchor

The complete 100-sample K32 semantic-chunk run (`recent=0`) scored 34/100 and
had ten `finish_reason=length` cases.  All ten truncated answers came from the
tool-history tasks and entered a tool-call/transcript continuation loop.  A
controlled four-sample cohort was chosen before tuning: two truncated/wrong
cases and two normally stopped/correct cases.  All runs used the same 32K
selection budget, 32K generation reserve, block size 32, 2K full-chunk query,
Adaptive Mean-K, FP8 KV, MTP-4, temperature 0.6, and seed 20260722.

| Recent anchor | Correct | Length | Per-sample result | Decision |
|---:|---:|---:|---|---|
| 0 (control) | 2/4 | 2/4 | 0, 1, 0, 1 | baseline |
| 128 tokens / 4 blocks | **3/4** | **0/4** | 1, 1, 0, 1 | retained candidate |
| 512 tokens / 16 blocks | **3/4** | **0/4** | 1, 1, 0, 1 | effective, but uses more budget |
| 1,024 tokens / 32 blocks | 0/2 | 0/2 | 0, 0 | stopped early; regression |
| 2,048 tokens / 64 blocks | 2/4 | 0/4 | 0, 1, 1, 0 | no net utility gain |

The effect is not monotonic in the amount of recent context.  A four-block
tail is enough to anchor the rendered dialogue/tool structure while consuming
only 0.39% of the K32 budget.  Reserving 1K or 2K displaces more semantic Top-K
blocks and causes different sample-level regressions.  On this cohort, 128 and
512 tokens produced identical final answers, so 128 is the preferred candidate.

Three additional formerly truncated Count Frequency (Tool) samples were
tested.  They all changed from `length` to a short, well-formed `stop`, but
remained incorrect.  Across the five formerly truncated samples tested in this
follow-up, only one recovered the correct answer.  The defensible conclusion is
therefore limited: a tiny recent anchor robustly improves generation stability
and gave a preliminary +1/4 accuracy gain, but it does not fix the underlying
retrieval/counting error in most truncated samples.  A full-100 run is required
before replacing the published K32 utility result.

An attempted chunk-local-only 2K pin (cleared before the final query) removed
the two observed length stops but did not recover either answer.  That engine
and API implementation was removed rather than retained.  The surviving
candidate uses the existing `--kvmem-recent-tokens` mechanism; the reproducible
launcher is:

- `scripts/kvmem_eval/run_agentlongbench_512k_k32_semantic_chunk_recent128.sh`

Primary artifacts:

- recent=128 four-sample cohort:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_k32_chunked_recent128_ab4_20260808`
- recent=512 four-sample cohort:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_k32_chunked_recent512_ab4_20260808`
- recent=2K four-sample negative control:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_k32_chunked_recent2k_ab4_20260808`
- chunk-local-only negative controls:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_k32_chunked_local2k_ab4_20260808`
  and
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_k32_chunked_local2k_ab2_20260808`

## Remaining irreducible cost under current semantics

The final run still spends about 201.6 s in the provisional phase (including
the roughly 50-s online exact scorer) and 159.3 s in replay. Chunked reselection
intentionally performs both:
the provisional pass constructs the context-conditioned retrieval query, while
the replay pass builds durable KV/index state under the selected window. Removing
either pass would change the algorithm. The inline/layer-streamed scorer has now
also verified that Q transfer was not the dominant end-to-end cost. Further
large gains require faster model-forward kernels or a semantic change such as a
less frequent reselection cadence; another local I/O copy optimization cannot
remove the remaining dual-forward floor.
