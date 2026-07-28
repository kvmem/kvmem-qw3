# KVMem CPU-only proactive write-back benchmark

Date: 2026-07-25

## Change

CPU-only `opt_2`/`opt_3` can now preserve an inclusive CPU copy before GPU
pressure:

```text
prefill chunk N completes
  -> packed V-page D2H into a pinned slab
  -> prefill chunk N+1 overlaps that D2H
  -> pinned slab scatters into the authoritative CPU tier asynchronously
  -> pressure eviction releases a clean GPU page without another D2H
```

The state transition is explicit:

```text
GPU only -> CPU copy in flight -> clean CPU backing
```

Pressure selection drains the bounded copy pipeline before treating a CPU slot
as clean.  Reset, reconfiguration, raw-K budget eviction, and query replay
already drain the same ownership queue.  CPU proactive write-back is enabled
only for capacity-qualified inclusive CPU tiers; smaller CPU caches retain the
previous pressure-time stage-out path.

## Controlled sample

Both runs used:

- AgentLongBench stable ID
  `eae5f30d87ba4d7ad6b631df6eb1bbfb3ab6eae5e3de221bb1135aaef55a28cc`;
- 517,353 prompt tokens;
- Qwen3.6-27B Q8_0 weights;
- FP16 KV;
- 200K KVMem context plus 32K generation reserve;
- block size 32, mean-K, immutable K, query replay, MTP-4;
- 64 GiB CPU tier, no SSD, `opt_3`;
- 2048-token prefill chunks.

## Result

| Metric | Existing CPU `opt_3` | Proactive CPU write-back | Change |
|---|---:|---:|---:|
| Aggregate pressure/final stage-out | 2786.303 ms | **50.181 ms** | **-98.2%** |
| Aggregate stage-in | 267.595 ms | 355.606 ms | +88.011 ms |
| Aggregate assembly | 1199.589 ms | 1300.456 ms | +100.867 ms |
| Model prefill / TTFT phase | 280.876 s | **279.464 s** | -1.412 s (-0.50%) |
| Prefill throughput | 1841.92 tok/s | **1851.23 tok/s** | +0.51% |
| Decode | 51.329 s | 51.485 s | within run variance |

The proactive run submitted 254 CPU copy batches.  Each full 2048-token chunk
copied 68 MiB:

- mean pinned-to-CPU scatter: 44.662 ms;
- maximum scatter: 48.585 ms;
- cumulative CPU-copy worker time: 11.344 s;
- cumulative exposed D2H wait: 7.564 ms.

The worker time is not request critical-path time: each approximately 45 ms
scatter ran during the following approximately 550--600 ms model-prefill
chunk.  At each of the first ten pressure selections, all 960 victims were
clean and approximately 0.996 GiB of D2H was avoided.  At final semantic
selection, all 3591 victims were clean and 3.726 GiB was avoided; its stage-out
was 12.916 ms.

The small stage-in/assembly differences are run-to-run variance and CPU-memory
bandwidth contention.  End-to-end prefill nevertheless improved by 1.412
seconds.  The primary result is removal of the multi-second pressure stall,
not a large total-prefill speedup: model compute still dominates.

## Correctness

- build passed;
- 12/12 CTest targets passed;
- prompt/completion token counts matched exactly;
- both runs generated the identical response
  `"\n\n<answer>1</answer>"`;
- both requests completed with `finish_reason=stop`;
- mean-K retrieval did not fall back.

This controlled sample is officially incorrect in both arms, so the run
validates output parity and performance rather than answer quality.

## Artifacts

- baseline log:
  `/data/chaidi/kvmem_eval/logs/agent512_generic_opt_baseline_fp16_cpu_opt3_smoke1_20260725_server.log`
- proactive log:
  `/data/chaidi/kvmem_eval/logs/agent512_cpu_proactive_fp16_opt3_smoke1_20260725_server.log`
- baseline result:
  `/data/chaidi/kvmem_eval/results/agent512_generic_opt_baseline_fp16_cpu_opt3_smoke1_20260725/`
- proactive result:
  `/data/chaidi/kvmem_eval/results/agent512_cpu_proactive_fp16_opt3_smoke1_20260725/`
