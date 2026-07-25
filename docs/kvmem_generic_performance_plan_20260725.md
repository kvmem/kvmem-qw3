# KVMem generic performance plan

Date: 2026-07-25

## Scope

This plan targets single-request KVMem execution and deliberately excludes
optimizations that only help a fixed benchmark schedule or one host topology:

- no cross-request/continuous-batching pipeline is required;
- no fixed CPU NUMA placement is compiled into the implementation;
- all SSD features are capability-gated and retain a portable fallback;
- correctness and selected block IDs must remain unchanged.

The current controlled AgentLongBench-512K FP16 baseline uses a 200K context
budget, 32K generation reserve, 32-token blocks, immutable source K, query
replay, MTP-4, 2048-token prefill chunks, a 64 GiB CPU tier, and no SSD.  A
representative 517K-token request spends roughly 290 seconds in model prefill,
2.6--2.8 seconds in pressure/final stage-out, 0.4--0.55 seconds in final
stage-in, and 1.1--1.3 seconds in raw-K assembly/re-RoPE.  This means transfer
optimizations can make reselection much faster, but only prefill compute
improvements can materially change end-to-end throughput.

## Required optimization stages

### 1. CPU-only proactive prefill write-back

After each completed prefill chunk, copy newly durable V pages to a pinned
double buffer.  The D2H for chunk N overlaps model compute for chunk N+1.
After its fence completes, scatter the slab into the CPU tier in a background
worker and retain the GPU pages until pressure selection actually evicts them.

At pressure time, a block with a valid CPU copy is clean: eviction only updates
the page table and releases the GPU page.  No new D2H is necessary.

The existing SSD write-back path is the structural baseline, but CPU-only
ownership and completion must not be represented as a fake SSD write.  The
feature is enabled only when the CPU tier can preserve the authoritative spill
record; otherwise the existing synchronous pressure path remains available.

Acceptance:

- exact output parity on a deterministic real sample;
- no lost slot on rollback, reconfiguration, or a partial final block;
- pressure stage-out bytes approach zero for blocks already copied;
- no increase in peak GPU memory beyond the pinned host buffers.

### 2. Fully incremental window assembly

Retain the existing `KvMemRemap::skip` rule and extend incremental behavior to
all assembly-derived state:

- rebuild K only for newly resident blocks or blocks whose compact position
  changed;
- reuse unchanged MTP K;
- update only changed page-table and remap metadata ranges when profitable;
- reuse or gather existing per-block mean-K data instead of recomputing the
  complete window when the retrieval/update mode permits it.

A compact-position change is a real RoPE change and cannot be skipped.  The
optimization must therefore report separate counts for incoming, moved, and
unchanged blocks.  It must not relabel a moved block as reusable merely because
its physical GPU page is unchanged.

Acceptance:

- bitwise parity for unchanged blocks;
- existing immutable-K and batched remap tests pass;
- assembly trace reports the expected changed/unchanged counts;
- reduced assembly time on a selection with high block retention.

### 3. Predictive prefetch during query replay

Before semantic scores are available, prefetch only blocks whose membership is
known:

- sink blocks;
- explicitly pinned/recent blocks;
- an optional bounded set of high-heat blocks from the prior selection epoch.

After scoring, merge the final selected IDs with the speculative set, cancel or
discard unused reads, and fetch only the remainder.  Prefetch must never change
ranking or the final selection.  CPU reads use the existing pinned gather/H2D
pipeline; SSD reads use the same ticket API as normal stage-in.

Acceptance:

- selected block IDs are identical with prefetch on/off;
- traces separate useful, unused, and late-prefetch blocks;
- exposed stage-in time falls without increasing total bytes excessively.

### 4. Native FP8 KV attention and assembly

The FP8 KV path currently stores raw e4m3 but the generic FlashInfer templates
convert values to FP16 inside attention.  Profile prefill, decode, append,
raw-K scatter, RoPE, and mean-K separately.  Replace only demonstrated
bottlenecks with kernels that consume/produce FP8 directly and fuse conversion
with the consuming operation rather than materializing an FP16 KV mirror.

The first target is the long-prefill paged-attention path; decode and assembly
are promoted only if profiling shows exposed time.  Numerical validation must
compare FP8 baseline and optimized FP8, not FP8 against FP16 bitwise.

Acceptance:

- no persistent FP16 KV mirror;
- FP8 output parity within the existing FP8 tolerance;
- no regression in retrieval fallback rate;
- recover the observed approximately 4.7% FP8 TTFT regression.

### 5. Portable SSD engine

Add a capability-selected backend:

1. calibrated `io_uring` with registered files/buffers and optional
   `O_DIRECT`;
2. batched `preadv`/`pwritev` worker-pool fallback;
3. current positional synchronous I/O as the correctness fallback;
4. optional GDS only when runtime probing succeeds.

Use fixed-offset records, coalesce consecutive block IDs, prioritize reads over
write-back, and prefetch predicted blocks.  Queue depth, slab size, direct-I/O
alignment, and concurrency are runtime tuning values, not machine constants.

This stage does not improve a 512K CPU-only run; it is required for contexts
whose durable records exceed available CPU memory.

## Additional generic optimizations

### Prefill-chunk autotuning

FP8 reduces KV pool pressure but the current explicit 2048-token chunk remains
fixed.  Benchmark 2048, 4096, and 8192 with the actual model, KV dtype, MTP
mode, and free-memory headroom.  Select the fastest safe size and cache the
choice by model/build/device configuration.  Explicit CLI values remain
authoritative.

This has the largest possible end-to-end impact because model prefill dominates
the current request.

### Incremental selection-derived metadata

Maintain a block-ID-to-window-slot map and dirty ranges for:

- main and MTP page tables;
- remap arrays;
- window block IDs/bases/token counts;
- per-window representative K when cumulative-attention scoring is enabled.

Use a full rebuild when the dirty fraction crosses a measured threshold; small
changes use incremental uploads.  This avoids making the sparse case slower
than the dense case.

### Kernel and synchronization reduction

Fuse raw-K scatter, FP8 conversion, and RoPE where they share the same input
and output.  Replace full-device synchronization used only for timing with
stream events.  Preserve synchronization needed for ownership or correctness.

### Optional GPU raw-K hot cache

A bounded GPU cache for frequently reselected raw-K blocks can avoid CPU gather
and H2D, but consumes memory that would otherwise increase the active KV
window.  It remains off by default and is tested only after the 48 GiB target
and FP8 active window are stable.

## Validation sequence

Each stage is committed and tested independently:

1. host-only store tests and CUDA assembly/FP8 unit tests;
2. deterministic small KVMem regression;
3. one fixed 517K AgentLongBench sample with performance traces;
4. FP16 A/B before changing dtype;
5. FP8 profiler and targeted A/B;
6. full FP8 AgentLongBench-512K 100-sample run;
7. fixed first 50 rows of the DeepseekMillion manifest for the 1M run.

The full runs keep raw answers, per-sample timing, token counts, selected block
traces, official evaluation rows, configuration, commit hash, and validation
reports in separate result directories.  Existing FP16 result directories are
never overwritten.
