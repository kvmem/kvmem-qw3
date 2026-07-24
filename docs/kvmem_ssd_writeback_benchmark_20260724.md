# KVMem SSD prefill write-through benchmark (2026-07-24)

## Scope

This benchmark validates the `opt_2` proactive prefill write-through pipeline
and the complete `opt_3` CPU+SSD stage-in pipeline. It is a matched one-sample
performance/correctness experiment, not an accuracy estimate.

All runs use the same AgentLongBench-Long sample:

```text
sample_id / trace tag:
eae5f30d87ba4d7ad6b631df6eb1bbfb3ab6eae5e3de221bb1135aaef55a28cc
prompt tokens: 517,353
```

Common model and KVMem settings:

```text
model: Qwen3.6-27B-Q8_0
logical ctx: 655,360
KVMem context budget: 204,800
generation reserve: 32,768
block size: 32
KV dtype: FP16
retrieval: mean-k
sink blocks: 8
query replay: on
immutable source K: on
MTP local positions: on
MTP chain: 4
prefill chunk: 2,048
CPU tier: 64 GiB
NVMe tier: 24 GiB on /dev/nvme0n1p3
I/O slab: 64 MiB
write queue depth: 8
GPU memory sampled during prefill: 47,080 MiB
```

`QW3_KVMEM_PERF_TRACE=1` was enabled for every run. This adds diagnostic
synchronization, so comparisons below are only between these instrumented
runs.

## End-to-end results

| Configuration | Prefill | Prefill throughput | Aggregate stage-out | Aggregate stage-in | Decode |
|---|---:|---:|---:|---:|---:|
| `opt_2`, write-through off | 287.800 s | 1,797.61 tok/s | 13,145.416 ms | 601.898 ms | 51.552 s |
| `opt_2`, proactive SSD only | 279.654 s | 1,849.97 tok/s | 91.314 ms | 5,753.996 ms | 51.415 s |
| `opt_2`, proactive SSD + async CPU admission | 279.524 s | 1,850.83 tok/s | 61.232 ms | 1,667.114 ms | 51.429 s |
| `opt_3`, full mixed CPU+SSD pipeline | **279.508 s** | **1,850.94 tok/s** | **60.956 ms** | **548.730 ms** | **51.278 s** |

Relative to the matched write-through-off baseline, the final `opt_3`
configuration:

- reduces prefill latency by 8.292 s (2.88%);
- raises prefill throughput by 2.97%;
- reduces measured stage-out time by 99.54%;
- makes stage-in 8.83% faster despite retaining the SSD tier;
- leaves decode throughput effectively unchanged.

The smaller end-to-end gain than the isolated stage-out reduction is expected:
proactive GPU gather/D2H consumes memory bandwidth while prefill kernels run.
The work is hidden as wall-clock waiting, but it is not free GPU bandwidth.

## Pressure-selection result

The baseline moved approximately 0.996 GiB at each 960-block pressure event:

```text
stage_out_ms: approximately 831 to 925 ms
stage_out_d2h_ms: approximately 68 ms
stage_out_clean_blocks: 0
```

With proactive write-through:

```text
stage_out_ms: approximately 4.5 to 4.8 ms
stage_out_d2h_ms: 0
stage_out_clean_blocks: 960 / 960
stage_out_clean_avoided_gib: 0.996
```

The old GPU pages are retained while write-through runs. Pressure selection
only releases them after a pinned D2H slab or clean SSD record is an independent
owner.

## Overlap evidence

For a typical 2,048-token chunk:

```text
record bytes: about 68 MiB
D2H exposed fence wait: 0.003 to 0.008 ms
time from D2H submit to next chunk boundary: about 545 to 568 ms
64 MiB CPU first-touch/copy: about 42 ms
CPU copy + SSD worker completion: about 124 to 136 ms
```

Both CPU admission and SSD persistence finish well inside the next prefill
chunk's compute window. Four pinned writeback slabs are used because one
68 MiB chunk spans two 64 MiB slabs; this provides two complete chunk-sized
buffer sets.

## Final semantic reselection

| Metric | Write-through off (`opt_2`) | Final mixed pipeline (`opt_3`) |
|---|---:|---:|
| Stage-out | 4,348.741 ms | 13.590 ms |
| Stage-in wall | 600.544 ms | 547.977 ms |
| CPU input | 3.552 GiB / 3,423 blocks | 3.552 GiB / 3,423 blocks |
| CPU gather | old per-block path | 501.698 ms |
| Packed H2D enqueue | old per-block path: 596.783 ms | 44.094 ms |
| Packed H2D fence wait | n/a | 0.461 ms |
| CPU H2D batches | 0 | 58 |
| Reselection total | 6,876.468 ms | 2,816.136 ms |

The final mixed path reduces this reselection's total latency by 59.05%.

An intermediate SSD-only run was important diagnostically: stage-in rose to
5.536 s because all 3.552 GiB had to be read from SSD. The production path
therefore feeds the existing heat-aware CPU cache from each completed D2H slab
before recycling it.

The benchmark also exposed and fixed an older `opt_3` condition that enabled
packed CPU stage-in only when no SSD tier existed. CPU hits and SSD misses can
now coexist: CPU records use packed gather/H2D and SSD records use coalesced
reads.

## Correctness

All four completed variants produced exactly the same raw answer:

```text
<answer>1</answer>
```

They also reported:

```text
query_expected=17
query_captured=17
requested=mean-k
used=mean-k
fallback=0
source_blocks=16,168
selected_blocks=6,400
completion_tokens=4,130
```

This single-sample parity verifies the storage pipeline did not change the
observed answer or silently degrade retrieval. Broader accuracy validation is
still required before treating it as a model-quality result.

## Artifacts

Baseline:

```text
/data/chaidi/kvmem_eval/logs/agent512_ssd_opt2_writeback_off_smoke1_20260724_server.log
/data/chaidi/kvmem_eval/results/agent512_ssd_opt2_writeback_off_smoke1_20260724/
```

SSD-only diagnostic:

```text
/data/chaidi/kvmem_eval/logs/agent512_ssd_opt2_writeback_on_smoke1_20260724_server.log
/data/chaidi/kvmem_eval/results/agent512_ssd_opt2_writeback_on_smoke1_20260724/
```

Async CPU-admission diagnostic:

```text
/data/chaidi/kvmem_eval/logs/agent512_ssd_opt2_writeback_cpu_async_smoke1_20260724_server.log
/data/chaidi/kvmem_eval/results/agent512_ssd_opt2_writeback_cpu_async_smoke1_20260724/
```

Final mixed pipeline:

```text
/data/chaidi/kvmem_eval/logs/agent512_ssd_opt3_mixed_pipeline_smoke1_20260724_server.log
/data/chaidi/kvmem_eval/results/agent512_ssd_opt3_mixed_pipeline_smoke1_20260724/
```

