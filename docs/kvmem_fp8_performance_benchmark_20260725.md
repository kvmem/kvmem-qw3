# KVMem FP8 performance benchmark

Date: 2026-07-25

## Scope

This benchmark profiles one fixed 517K-token AgentLongBench sample after the
CPU-only proactive write-back and incremental MTP assembly changes.  Every run
uses:

- Qwen3.6-27B-Q8_0;
- 200K KVMem context budget plus 32K generation reserve;
- 32-token blocks, mean-K retrieval, eight sink blocks, no sub-blocks;
- immutable source K, query replay, MTP-4, `opt_3`;
- a 64 GiB CPU tier and no SSD;
- the same prompt, seed, temperature, thinking budget, and 32K maximum output.

The benchmark sample is
`eae5f30d87ba4d7ad6b631df6eb1bbfb3ab6eae5e3de221bb1135aaef55a28cc`.
All measured FP8 variants produced the same final hypothesis
`<answer>1</answer>`.  The reference answer is zero, so this sample is a
performance/parity control rather than evidence of FP8 task accuracy.

## Prefill chunk result

| KV dtype | Prefill chunk | Prefill | Throughput | Decode throughput | Observed GPU peak |
|---|---:|---:|---:|---:|---:|
| FP16 | 2048 | 279.686 s | 1849.77 tok/s | 77.40 tok/s | about 47.1 GiB |
| FP8 | 2048 | 294.823 s | 1754.79 tok/s | 83.75 tok/s | about 39.9 GiB |
| FP8 | 4096 | 277.012 s | 1867.62 tok/s | 84.76 tok/s | about 41.0 GiB |
| FP8 | 8192 | 273.418 s | 1892.17 tok/s | 86.49 tok/s | about 43.3 GiB |

Raw FP8 at the old 2048 chunk is 5.41% slower in prefill than FP16.  Raising
the FP8 chunk to 4096 recovers that regression; 8192 improves FP8-2048 by
7.26% and is 2.24% faster than FP16-2048 while remaining below the 48 GiB
limit.  The 4096-to-8192 gain is only 1.30%, so 8192 is the selected upper
point; a 16K chunk was not attempted because the extrapolated memory peak is
too close to the hard limit for a small expected gain.

An 8192-token chunk writes about 135.5 MiB of immutable spill data and is
therefore split across the default 128 MiB slabs.  The split exposes roughly
80--90 ms while outstanding CPU copies are drained at a pressure boundary, but
the cumulative cost is small compared with model prefill.

Artifacts:

- FP8-2048:
  `/data/chaidi/kvmem_eval/results/agent512_fp8_opt3_chunk2048_smoke1_20260725`
- FP8-4096:
  `/data/chaidi/kvmem_eval/results/agent512_fp8_opt3_chunk4096_smoke1_20260725`
- FP8-8192:
  `/data/chaidi/kvmem_eval/results/agent512_fp8_opt3_chunk8192_smoke1_20260725`

The corresponding server logs use the same basename under
`/data/chaidi/kvmem_eval/logs/` with `_server.log` appended.

## Incremental assembly

The first FP8 pressure selection builds the whole 6400-block MTP window.
Subsequent pressure selections rebuild only incoming MTP blocks and re-RoPE
retained moved blocks in place:

| Chunk | First assembly | Steady assembly | New MTP blocks | In-place MTP blocks |
|---:|---:|---:|---:|---:|
| 2048 | 139.4 ms | 14--17 ms | 960 | 5432 |
| 4096 | 89.9 ms | about 7.8 ms | 896 | 5496 |
| 8192 | 97.1 ms | about 7.8 ms | 1020 | 5372 |

The exact incoming count changes with chunk/pressure alignment.  Selected
block ranking is unchanged; the optimization only changes construction of the
already selected working K.

## FlashInfer plan reuse

The paged prefill scheduler previously rebuilt and uploaded the same plan once
per normal-attention layer.  The cache now creates one plan for a
`(workspace, stream, batch, page count, heads, head dimension, page size)`
shape and reuses it for the remaining 15 layers in that prefill chunk.
Cache-hit layers also avoid fencing an acquired host plan buffer that they did
not use.

The first cache-only complete run was 273.174 s versus 273.418 s without the
cache, a 0.244 s (0.09%) improvement.  Per-chunk trace shows 15 hits for every
one miss, but scheduler construction is not a material end-to-end bottleneck.
The optimization is retained because it is bounded, output-identical, and has
no persistent memory cost.

Artifact:

`/data/chaidi/kvmem_eval/results/agent512_fp8_opt3_chunk8192_plancache_full1_20260725`

## Query-first-pass prefetch

The original implementation was accidentally wired only into the non-MTP
generation path.  After wiring the same operation into MTP query replay, the
fixed sample reports:

- 410 speculative CPU blocks loaded into free generation-reserve pages;
- 213 of those blocks present in the final semantic selection;
- 51.95% hit rate;
- 144.37 ms speculative interval, fully overlapped with first-pass query
  computation;
- ordinary semantic CPU stage-in reduced from 3194 to 2981 blocks;
- total transferred stage-in blocks increased from 3194 to 3391 because 197
  speculative blocks were unused;
- semantic reselect remained essentially unchanged at about 374 ms.

The ordinary stage-in was already hidden by approximately 324 ms of raw-K
assembly.  Speculation therefore did not reduce exposed TTFT and increased
CPU/PCIe traffic by 6.17%.  The feature remains available through
`QW3_KVMEM_QUERY_PREFETCH=1`, but is disabled by default.

Artifact:

`/data/chaidi/kvmem_eval/results/agent512_fp8_opt3_chunk8192_prefetch_plancache_final1_20260725`

## Native fused FP8 status

The current generic FlashInfer kernel reads raw E4M3 K/V and converts values in
the consuming attention kernel; it does not allocate a persistent FP16 KV
mirror.  An experimental adapter attempted to instantiate FlashInfer's native
FP8 GMMA/TMA paged prefill.

It was rejected from the implementation:

- the installed implementation is Hopper-specific and cannot compile for the
  current Blackwell `sm_120a` GPU;
- an explicit offline `sm_90a` compile also fails for the required E4M3
  head-dimension 128/256 configurations with unsupported GMMA/TMA
  instantiations;
- Blackwell FlashInfer exposes a contiguous FP8 FMHA path, but gathering a
  complete 200K paged window per layer would cost more than the conversion it
  is meant to remove.

A true native paged FP8 path therefore requires a separately developed and
validated architecture-specific kernel.  No unbuildable adapter or dormant
architecture branch is committed.

## Accuracy control: 200K versus 224K

The performance microbenchmarks above intentionally used a 200K selected
context so they could compare dtype and chunk size below a fixed 48 GiB GPU
limit.  That budget must not be treated as the production accuracy setting.
An initial FP8 full-run attempt exposed a large retrieval-budget effect:

| Configuration | Evaluated subset | Correct | `finish_reason=length` |
|---|---:|---:|---:|
| FP8, 200K, chunk 8192, current binary | first 8 | 2/8 | 2/8 |
| FP16, 224K, earlier full-run artifact | same first 8 | 6/8 | 0/8 |

The following matched controls isolate the cause:

- FP16, 200K, current binary: the third sample repeats tool calls until the
  32K generation limit and is incorrect.
- FP16, 224K, current binary: the same third sample stops normally and is
  correct.
- FP8, 224K, current binary: the three sensitive samples at original
  positions 3, 4, and 5 are all correct and stop normally.

Therefore the regression is caused by reducing the semantic selection budget
from 224K to 200K, not by FP8, the 8192-token prefill chunk, or the current
performance optimizations.  The incomplete 200K run is preserved as a budget
ablation and is not resumed.

Artifacts:

- FP8 200K, first eight before the run was stopped:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_normal100_k200k_g32k_b32_qr_immutable_mtp4_fp8_cpu64_opt3_full100_v2_20260725`
- FP16 200K, first three:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_ab_fp16_k200k_chunk8192_opt3_first8_20260725`
- FP16 224K, sensitive sample 3:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_ab_fp16_k224k_chunk8192_opt3_sample3_20260725`
- FP8 224K, sensitive samples 3, 4, and 5:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_ab_fp8_k224k_chunk8192_opt3_key3_20260725`

## Cold-request host allocation reuse

Independent requests used to release all demand-allocated sparse CPU slabs and
immutable raw-K chunks, then call `malloc_trim(0)`.  The next long request
would fault and rebuild the same bounded allocation topology.  `opt_3` now
invalidates all block mappings, live-slot flags, and raw-K validity while
retaining only the backing allocations.  The retained physical bytes remain
strictly bounded by `--kvmem-cpu-gb`; raw-K growth can reclaim an empty retained
spill slab.  `QW3_KVMEM_RETAIN_COLD_HOST=0` restores minimum-idle-RSS behavior.

A direct long-request/short-request reset probe measured:

| Cold reset mode | Total | Tier clear | Host release/trim |
|---|---:|---:|---:|
| release allocations | 1121.602 ms | 574.564 ms | 546.149 ms |
| retain invalid buffers | 2.373 ms | 1.089 ms | 0.477 ms |

The matched two-sample AgentLongBench run preserved both outputs and both
official scores.  On the second sample, content-first latency fell from
334.138 to 330.947 seconds, while native prefill varied from 284.827 to
282.518 seconds.  The direct reset probe is therefore the reliable isolated
effect: about 1.119 seconds per independent long request, roughly 0.3% of a
512K request.  This is a small but generic allocator-churn reduction, not the
previously suspected 49-second bottleneck.

Artifacts:

- two-sample retained-buffer A/B:
  `/data/chaidi/kvmem_eval/results/agentlongbench_512k_ab_fp8_k224k_chunk8192_opt3_retainhost_first2_20260725`
- retained reset phase log:
  `/data/chaidi/kvmem_eval/logs/agentlongbench_512k_cold_reset_diag_fp8_k224k_opt3_20260725_server.log`
- release/trim reset phase log:
  `/data/chaidi/kvmem_eval/logs/agentlongbench_512k_cold_reset_diag_fp8_k224k_opt3_noretain_20260725_server.log`

## TTFT metric correction

The AgentLongBench runner previously documented `ttft_sec` as the first
streaming delta but implemented a content-first choice: if final-answer content
eventually appeared, it ignored an earlier reasoning delta.  For a sample with
46.8 seconds of streamed reasoning, this made `ttft_sec` look about 48 seconds
larger than native prefill even though the first reasoning token arrived at the
expected time.

New runs define `ttft_sec` as the earliest non-empty token across the
`reasoning_content` and `content` channels.  `content_ttft_sec` and the existing
`first_content_sec` retain answer-visible latency, while
`first_reasoning_sec` remains separately available.  Historical result files
are not rewritten; when comparing old results, use the minimum of their
`first_reasoning_sec` and `first_content_sec`.

## Selected production profile

For the full FP8 AgentLongBench runs:

- `--kv-dtype fp8`;
- `--prefill-chunk 8192`;
- `--kvmem-budget 229376` (224K context) plus a 32K generation reserve;
- `--kvmem-optimization-level opt_3`;
- plan reuse enabled (default);
- query speculative prefetch disabled (default);
- CPU tier preferred; SSD used only if the logical source exceeds safe host
  capacity.

At 200K, the profile preserves roughly 4.7 GiB GPU headroom below the 48 GiB
target on the fixed sample.  The accuracy-preserving 224K FP8 control remains
below 48 GiB as well, with an observed peak of about 43--44 GiB.
