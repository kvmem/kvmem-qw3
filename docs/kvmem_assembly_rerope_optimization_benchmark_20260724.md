# KVMem assembly / re-RoPE optimization

Date: 2026-07-24

## Problem

On the 517K-token AgentLongBench smoke sample, the final semantic
reselection took 2816.136 ms.  Of that, assembling the selected 200K-token
working set and re-applying RoPE took 2031.572 ms.

The original path had two independent costs:

1. CUDA remap kernels repeatedly evaluated `powf` and `sincosf`.
2. Immutable raw K was gathered, copied to the GPU, and scattered in
   synchronous batches.  MTP used 50 small batches for 6400 selected blocks.

## Implementation

The optimized implementation contains two layers:

- `table`: build a model-sized FP32 `[position, pair, sin/cos]` table once
  and use it in the in-place and immutable raw-K materialization kernels.
  The table is 64 MiB for this model (`262144 * 32 * 2 * sizeof(float)`).
- `pipeline`: use two pinned host buffers and two GPU staging buffers to
  overlap CPU gather for batch N+1, H2D for batch N, and GPU scatter/RoPE for
  batch N-1.  Cross-stream events protect buffer reuse.  MTP reuses these
  byte-sized buffers and increases its batch size from 128 to 2048 blocks,
  reducing 50 transfers to 4.

The table kernels intentionally preserve the legacy de-rotate/re-rotate
arithmetic order.  CUDA parity tests report zero bit mismatches.

`opt_3` enables the full pipeline by default.  The environment variable
`QW3_KVMEM_ASSEMBLY_MODE=legacy|table|pipeline` provides explicit A/B and
rollback modes.

## Results

All runs used the same 517353-token AgentLongBench prompt, 200K KVMem
context budget, 32K generation reserve, 32-token blocks, mean-K retrieval,
immutable K, query replay, MTP, and 2048-token chunked prefill.

| Metric | Legacy baseline | Table only | Pipeline | Pipeline vs baseline |
|---|---:|---:|---:|---:|
| Final semantic assembly / re-RoPE | 2031.572 ms | 1932.171 ms | 686.739 ms | -66.2%, 2.96x |
| Final semantic reselection total | 2816.136 ms | 2725.985 ms | 1464.839 ms | -48.0%, 1.92x |
| Assembly across all 12 calls | 3829.138 ms | 3594.063 ms | 1434.816 ms | -62.5%, 2.67x |
| Full prompt prefill | 279.508 s | 279.103 s | 278.529 s | -0.979 s |
| Prefill throughput | 1850.94 tok/s | 1853.63 tok/s | 1857.45 tok/s | +0.35% |
| Peak GPU memory | about 47080 MiB | 47178 MiB | 47426 MiB | +346 MiB |

The pipeline final semantic assembly processed 4.238 GiB of raw K in 35
batches.  Only 2.098 ms of H2D wait remained exposed on the critical path.
The raw-gather timing was 679.609 ms, so CPU raw-K gathering is now the
dominant assembly cost rather than PCIe transfer or GPU RoPE.

The raw answer was identical in all three runs:

```text
<answer>1</answer>
```

This particular smoke sample is incorrect under the official judge in every
mode, so it validates output parity and performance rather than an accuracy
gain.

## Correctness checks

- `cmake --build build -j4`: passed.
- `ctest --test-dir build --output-on-failure`: 12/12 passed.
- `qw3-rope-remap-batched-parity`: table path has `bit_mismatches=0`.
- `qw3-kvmem-immutable-k`: passed; table raw scatter is bitwise equal to
  direct immutable-K bake.
- Real 517K-token integration run: no retrieval fallback and identical output.

## Artifacts

- Baseline log:
  `/data/chaidi/kvmem_eval/logs/agent512_ssd_opt3_mixed_pipeline_smoke1_20260724_server.log`
- Table log:
  `/data/chaidi/kvmem_eval/logs/agent512_assembly_table_smoke1_20260724_server.log`
- Pipeline log:
  `/data/chaidi/kvmem_eval/logs/agent512_assembly_pipeline_smoke1_20260724_server.log`
- Table result directory:
  `/data/chaidi/kvmem_eval/results/agent512_assembly_table_smoke1_20260724/`
- Pipeline result directory:
  `/data/chaidi/kvmem_eval/results/agent512_assembly_pipeline_smoke1_20260724/`

## Follow-up decision

A direct-delta RoPE kernel was considered but not promoted.  After the table
and pipeline changes, exposed H2D wait is about 2 ms and non-gather GPU work
is only a few milliseconds on steady-state pressure assemblies.  Direct
delta would therefore offer little end-to-end gain while changing numerical
behavior.  The next meaningful optimization target is the CPU raw-K gather
layout and worker scheduling.
