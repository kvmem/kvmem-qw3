# KVMem Adaptive index optimization (2026-07-30)

## Scope

This work optimizes both Adaptive Multi-Prototype index placements without
changing retrieval semantics:

- GPU placement: keep the complete Adaptive index resident on GPU.
- CPU placement: keep the canonical index in pageable host memory and use the
  GPU only for online dot/softmax/MaxSim scoring.

The retained compatibility paths are:

```text
QW3_KVMEM_ADAPTIVE_GPU_PACKED=1
--kvmem-adaptive-score-mode tiled-two-pass
```

## Implemented changes

### 1. CPU layer-one-pass streaming

The old bounded CPU path transferred every prototype twice:

1. pass 1 built the exact softmax LSE;
2. pass 2 recomputed the same dots to reduce per-block MaxSim mass.

The new path stages one complete normal-attention layer, stores a bounded
logits batch, and finishes the exact layer contribution before reusing the
slot. It therefore transfers the index once and evaluates each prototype dot
once. Two layer-sized slots overlap pageable-to-pinned packing, H2D, and GPU
scoring.

Modes:

```text
--kvmem-adaptive-score-mode auto
--kvmem-adaptive-score-mode layer-one-pass
--kvmem-adaptive-score-mode tiled-two-pass
```

`auto` uses layer-one-pass when the largest layer is at most 256 MiB and
records an explicit fallback reason otherwise.

### 2. GPU per-layer incremental arenas

The old GPU path waited until the final query to gather and upload a 3.445 GiB
global packed image. The new path:

- retains appendable host vectors as the canonical index;
- maintains one growable GPU arena per normal-attention layer;
- uploads only the dirty suffix after each prefill capture;
- overlaps those uploads with later prefill work;
- finalizes only small block-offset/count metadata at query time;
- scores the resident per-layer arenas with the same layer-one-pass scorer.

The final-query index finalization no longer copies the full index.

### 3. GQA group-6 specialization

Qwen3.6-27B uses `gqa_group=6`. Packed, CPU-streamed, and per-layer Adaptive
scorers now load a K prototype once and form all six associated query-head
dots. The CUDA compiler reports zero spill stores and zero spill loads for the
group-6 specializations.

## Correctness

`qw3-kvmem-softmax-pages` now tests the production group-6 shape. All paths
match the CPU reference:

| Comparison | Maximum absolute error |
|---|---:|
| packed vs host | 5.588e-9 |
| two-pass streamed vs packed | 5.588e-9 |
| two-pass streamed vs host | 3.725e-9 |
| layer-one-pass vs packed | 5.588e-9 |
| layer-one-pass vs host | 3.725e-9 |

The final 1M A/B selected the same 448 blocks in both placements:

```text
intersection = 448
union = 448
fallback = 0
```

## Controlled 1M A/B

Artifacts:

```text
/data/chaidi/kvmem_eval/ab/adaptive_index_placement_1m_20260730_1900
```

Configuration:

- AgentLongBench-1M question
  `2167e18164c2bfeffc68feb4ebc9e7a66204e04c167289a9643cfcebc267f021`
- 1,033,841 prompt tokens
- Qwen3.6-27B-Q8_0
- context limit 1,310,720
- KVMem budget 229,376 (224K)
- generation reserve 32,768
- retrieval block 512 tokens; scoring slice 32 tokens
- Adaptive gains 0.10 / 0.06
- FP8 KV; FP16 Adaptive index/query
- query replay, immutable source K, MTP chain 4
- generation limited to one token to isolate TTFT/reselection

### Final optimized placements

| Metric | GPU incremental | CPU layer-one-pass |
|---|---:|---:|
| Server-ready VRAM | 37,650 MiB | 37,650 MiB |
| Peak process VRAM | 45,502 MiB | 41,868 MiB |
| Request VRAM delta | 7,852 MiB | 4,218 MiB |
| CPU placement VRAM saving | - | **3,634 MiB** |
| Index authority | 3.445 GiB | 3.445 GiB |
| GPU-resident index | 3.445 GiB | 0 |
| Prefill | 671.936 s | 671.015 s |
| Prefill throughput | 1,538.60 tok/s | 1,540.71 tok/s |
| Final index finalize | **0.176 ms** | 0.001 ms |
| Adaptive score wrapper | **173.414 ms** | 765.066 ms |
| CPU stream core | - | **453.733 ms** |
| CPU index H2D | - | **3.432 GiB** |
| CPU pageable-to-pinned pack | - | 422.727 ms |
| CPU H2D wait | - | 0.085 ms |
| Semantic reselection | **591.775 ms** | 1,268.399 ms |
| End-to-end TTFT | 676.696 s | 676.431 s |

The 0.265 s TTFT difference is not evidence that CPU scoring is faster:
prefill alone differed by 0.921 s between the two sequential runs. For
multi-turn operation, isolated scoring and reselection are the relevant
numbers.

### Improvement over the previous implementation

| Metric | Previous GPU | New GPU | Previous CPU | New CPU |
|---|---:|---:|---:|---:|
| Peak process VRAM | 44,964 MiB | 45,502 MiB | 41,500 MiB | 41,868 MiB |
| Prefill | 671.951 s | 671.936 s | 670.831 s | 671.015 s |
| Index finalize | 513.933 ms | **0.176 ms** | 0.001 ms | 0.001 ms |
| Score wrapper | 694.548 ms | **173.414 ms** | 1,555.895 ms | **765.066 ms** |
| Stream transfer | - | - | 6.863 GiB | **3.432 GiB** |
| Stream core | - | - | 1,482.345 ms | **453.733 ms** |
| Semantic reselection | 1,163.442 ms | **591.775 ms** | 2,502.249 ms | **1,268.399 ms** |
| TTFT | 677.249 s | 676.696 s | 677.505 s | 676.431 s |

Interpretation:

- GPU finalization fell by 99.97%; semantic reselection fell by 49.1%.
- CPU transfer volume was halved; stream core improved 3.27x; score wrapper
  fell by 50.8%; semantic reselection fell by 49.3%.
- Prefill throughput is unchanged within run-to-run noise.
- CPU one-pass needs two approximately 247 MiB staging slots, explaining its
  roughly 368 MiB peak increase over the old 64 MiB-slot path.
- Per-layer 64 MiB GPU arena rounding gives 4.000 GiB capacity for a 3.445 GiB
  index, explaining the roughly 538 MiB GPU peak increase.

## Should scoring run on CPU?

No, not as the general online path.

For this sample, the final query has 23 tokens. A pure CPU scorer would have
to scan approximately:

```text
3.445 GiB * 23 = 79.2 GiB
```

of prototype rows while also performing FP16 conversion, six query-head dots
per KV head, softmax, and block MaxSim. A 256-token query raises the minimum
index traffic to approximately 882 GiB.

The optimized CPU-placement run reports only 0.085 ms of H2D wait because H2D
is already hidden behind host packing and GPU scoring. Moving scoring to CPU
therefore does not remove a meaningful PCIe critical path; it competes with KV
stage-in and writeback for the same CPU cores and memory channels instead.

The practical split remains:

- CPU: canonical capacity, prototype metadata, packing, prefetch.
- GPU: dot products, softmax, MaxSim reduction.

## Placement recommendation

- Use `gpu` for repeated/multi-turn queries when an additional 3.6 GiB of VRAM
  is available. Its isolated semantic reselection is about 2.14x faster.
- Use `cpu` when VRAM capacity or colocated experiments matter more. It saves
  3,634 MiB in the measured 1M workload while preserving exactly the selected
  blocks.
- Keep `auto` score mode. Force `tiled-two-pass` only for compatibility or
  when a layer exceeds the bounded one-pass staging cap.

## Remaining optimization opportunities

1. GPU incremental growth uploaded 9.435 GiB in total because a growing arena
   must republish its existing prefix after reallocation. The cost was spread
   across 671.9 s of prefill and caused no measured throughput regression, but
   predictive per-layer capacity can reduce this traffic.
2. CPU one-pass is now dominated by the 422.7 ms pageable-to-pinned copy, not
   PCIe wait. The GPU is local to NUMA node 0; a future portable affinity policy
   should discover the GPU-local node at runtime rather than hard-code this
   machine's CPU list.
3. Pinning the complete 3.445 GiB host index would remove the copy but is not a
   practical default: it consumes a large locked-memory budget and scales
   poorly toward 10M contexts.
