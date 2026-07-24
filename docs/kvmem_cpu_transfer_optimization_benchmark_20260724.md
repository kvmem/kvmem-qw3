# KVMem CPU-only Opt2/Opt3 benchmark (2026-07-24)

## Scope

This benchmark isolates the CPU-only transfer path added to the cumulative
storage profiles:

- `opt_2`: two pinned 64 MiB D2H slabs, GPU page gather, one contiguous D2H
  per slab, CPU scatter, and batched GPU page-table release;
- `opt_3`: four-thread CPU gather into two pinned 64 MiB slabs, one contiguous
  H2D per slab, and GPU scatter into arbitrary physical KV pages;
- sparse immutable-K spill slots: lazy 64 MiB pageable slabs rather than one
  large allocation per block.

The CUDA backend uses one reusable 64 MiB GPU staging slab for both directions.
The CPU tier remains exclusive when no SSD is configured. Retrieval ranking,
selected block IDs, FP16 KV values, query replay, and immutable source K are
unchanged.

## Controlled sample

All rows use the same AgentLongBench-512K sample:

- stable sample ID:
  `eae5f30d87ba4d7ad6b631df6eb1bbfb3ab6eae5e3de221bb1135aaef55a28cc`;
- prompt: 517,343 tokens;
- task: `Count Frequency(Tool)`;
- model: `Qwen3.6-27B-Q8_0.gguf`;
- context window: 640K;
- KVMem: 200K context + 32K generation reserve, block size 32;
- mean-k, query replay, immutable source K, MTP-4, FP16 KV;
- CPU tier: 64 GiB; NVMe disabled;
- prefill chunk: 2,048;
- thinking budget: 4,096; temperature 0.6; top-p 0.95.

The final semantic selection loaded 3,423 blocks / 3.552 GiB and staged out
3,591 blocks / 3.726 GiB in every profile.

## Results

Aggregate timings cover all ten pressure selections plus the final semantic
selection:

| Implementation | Aggregate stage-out | Aggregate stage-in | TTFT |
|---|---:|---:|---:|
| prior CPU path | 9,207.6 ms | 1,583.1 ms | 344.323 s |
| Opt2, batched slabs/fences | 9,006.2 ms | 1,528.9 ms | 343.749 s |
| Opt3, 4-thread CPU gather | 8,979.6 ms | 787.9 ms | 343.196 s |
| Opt3, packed H2D + GPU scatter | 9,010.0 ms | 473.3 ms | 342.561 s |
| final, packed D2H/H2D + sparse slabs | **8,340.9 ms** | **359.1 ms** | 342.564 s |

Against the prior CPU path, the final implementation reduces aggregate
stage-in by 77.3% and aggregate stage-out by 9.4%. TTFT improves by 1.76
seconds (0.5%); full model prefill and decode dominate the approximately
342-second request, so sub-second run-to-run compute variance is visible in
the end-to-end number.

The final semantic reselection gives the clearest transfer-controlled
comparison:

| Metric | Opt2 batched baseline | Final Opt3 | Change |
|---|---:|---:|---:|
| CPU stage-in wall | 1,528.1 ms | **358.2 ms** | -76.6% |
| CPU gather | eager/per-page | 261.6 ms | n/a |
| H2D enqueue | 1,517.8 ms | **43.5 ms** | -97.1% |
| exposed H2D wait | 0.0 ms | 0.5 ms | effectively hidden |
| stage-out wall | 2,449.3 ms | **2,279.4 ms** | -6.9% |
| D2H submit | 188.6 ms | **6.0 ms** | -96.8% |
| CPU scatter | 266.6 ms | 300.3 ms | +12.6% |
| complete semantic reselection | 6,074.7 ms | **4,495.7 ms** | -26.0% |

Stage-out wall time improves less than D2H submission because the timer also
contains synchronization with already-queued prefill work, CPU destination
scatter, tier bookkeeping, and GPU page reuse. The copy-stream fence itself
exposed only 0.53 ms across 60 D2H batches, so the continuous DMA was already
hidden behind host preparation.

## Correctness and memory

- all 12 CTest targets passed;
- mean-k used the tiled two-dot scorer with `fallback=0`;
- expected and captured query lengths were both 17 tokens;
- the final output remained exactly
  `raw_response="\n\n<answer>1</answer>"`, with the same reasoning and token
  usage as the matched pre-optimization runs;
- sampled GPU use under full load was 47,114 MiB, 2,038 MiB below 48 GiB;
- no OOM, transfer error, or validation error occurred.

The task's official score is still zero because the model answer itself is
wrong for this sample. This is a transfer correctness/performance comparison,
not an accuracy claim.

## Artifacts

- prior CPU path:
  `/data/chaidi/kvmem_eval/logs/agentlongbench_512k_k200k_memfix_smoke_20260724_server.log`
- Opt2:
  `/data/chaidi/kvmem_eval/logs/agent512_cpu_transfer_opt2_k200k_smoke1_20260724_server.log`
- final Opt3:
  `/data/chaidi/kvmem_eval/logs/agent512_cpu_transfer_opt3_final_k200k_smoke1_20260724_server.log`
- final answer/evaluation:
  `/data/chaidi/kvmem_eval/results/agent512_cpu_transfer_opt3_final_k200k_smoke1_20260724/`

