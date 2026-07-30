# KVMem Adaptive index placement A/B (2026-07-30)

## Purpose

Measure the memory and latency trade-off between keeping the dynamically
packed Adaptive Multi-Prototype Mean-K index on GPU and keeping its authority
on CPU with bounded GPU streaming.

The A/B changes only:

```text
--kvmem-index-placement gpu|cpu
```

## Controlled configuration

- Dataset: AgentLongBench-1M
- Question ID:
  `2167e18164c2bfeffc68feb4ebc9e7a66204e04c167289a9643cfcebc267f021`
- Actual prompt: 1,033,841 tokens
- Model: Qwen3.6-27B-Q8_0
- Context limit: 1,310,720
- KVMem context budget: 229,376 tokens
- Generation reserve: 32,768 tokens
- Physical/retrieval block: 512 tokens
- Scoring slice: 32 tokens
- Retrieval: key-direction-adaptive
- Adaptive thresholds: 1→2 = 0.10, 2→4 = 0.06
- KV dtype: FP8
- Adaptive index/query dtype: FP16
- Query replay: enabled
- Immutable source K: enabled
- MTP chain: 4
- CPU index staging: two 64 MiB slots
- Generation capped at one token to isolate TTFT and retrieval cost

Raw artifacts:

```text
/data/chaidi/kvmem_eval/ab/adaptive_index_placement_1m_20260730_1616
```

## Result

| Metric | GPU-resident index | CPU-offloaded index |
|---|---:|---:|
| Server-ready VRAM | 37,650 MiB | 37,650 MiB |
| Peak process VRAM | 44,964 MiB | 41,500 MiB |
| Request-time VRAM delta | 7,314 MiB | 3,850 MiB |
| VRAM saved by CPU placement | - | **3,464 MiB** |
| Adaptive index authority | 3.445 GiB | 3.445 GiB |
| GPU-resident Adaptive index | 3.445 GiB | 0 |
| Prefill | 671.951 s | 670.831 s |
| Prefill throughput | 1,538.56 tok/s | 1,541.13 tok/s |
| Index finalize | 513.933 ms | 0.001 ms |
| Adaptive scoring wrapper | 694.548 ms | 1,555.895 ms |
| CPU streaming core | - | 1,482.345 ms |
| CPU index bytes transferred | - | 6.863 GiB |
| Semantic reselection total | 1,163.442 ms | 2,502.249 ms |
| End-to-end TTFT | 677.249 s | 677.505 s |
| Retrieval fallback | 0 | 0 |

Both arms selected exactly the same 448 blocks:

```text
intersection = 448
union = 448
```

The CPU path therefore preserves the retrieval result in this experiment.
Its extra latency is an index-placement cost, not a ranking change.

## Breakdown

### GPU-resident path

The 513.933 ms finalization contains:

```text
pageable host → pinned gather: 422.487 ms
H2D wait:                         2.305 ms
pipeline total:                428.440 ms
```

The PCIe copy is already almost completely hidden. The dominant finalization
cost is gathering the per-layer host vectors into the contiguous pinned upload
slabs. Subtracting finalization from the 694.548 ms scoring wrapper leaves
approximately 180.6 ms for metadata/scoring work.

### CPU-offloaded path

The exact two-pass streaming path reports:

```text
index transferred: 6.863 GiB
pageable → pinned pack: 684.802 ms
H2D backpressure/wait: 718.381 ms
streaming total:      1482.345 ms
```

The 3.445 GiB index is transferred twice because pass 1 builds the exact
softmax denominator and pass 2 computes per-block MaxSim mass. CPU packing and
H2D backpressure dominate the streaming time.

The selected-KV stage-in/assembly was also slower in the CPU arm
(929.090 ms versus 450.391 ms assembly wall time). Because selected blocks are
identical, this is not a retrieval difference. It may be run-to-run NUMA or
host-memory-bandwidth variability, or interference immediately after scanning
the full CPU index twice. It should be profiled separately before attributing
it to the placement algorithm.

## Optimization priority

### P0: CPU single-transfer, single-dot layer streaming

Softmax is independent between normal-attention layers. Stage one complete
layer (maximum roughly 250 MiB for this sample), run a one-dot layer scorer,
accumulate its normalized contribution, then reuse the staging allocation for
the next layer.

Expected effect:

- transfer 3.445 GiB instead of 6.863 GiB;
- eliminate the second prototype dot pass;
- retain roughly 3.0–3.2 GiB of the current VRAM saving, depending on whether
  one or two approximately 256 MiB staging slots are used.

### P1: GPU incremental per-layer index construction

Do not gather and upload all 3.445 GiB at the final query boundary. Maintain
per-layer GPU arenas and append newly finalized prototypes during chunked
prefill, overlapping the transfer with later prefill compute.

Expected effect:

- remove most of the 513.933 ms final-query index finalization;
- preserve the low-latency GPU-resident repeated-query path.

### P2: support GQA group 6

The real model reports `gqa_group=6`; current specialized Adaptive kernels only
fuse groups 2, 3, and 4. Add a group-6 specialization to both GPU-resident and
CPU-streaming scorers. This targets compute and K-row reads but does not remove
the dominant CPU double transfer or GPU host gather, so it follows P0/P1.

### P3: host-memory/NUMA follow-up

Profile the CPU index pack and subsequent raw-K gather by NUMA node and worker
affinity. Consider per-layer pinned or registered slabs only after P0 halves
the required traffic; pinning the entire multi-GiB index is not the default
solution.

## Practical choice

- Prefer `gpu` when repeated queries reuse a stable index and 3.5 GiB of VRAM
  is available.
- Prefer `cpu` for larger contexts or colocated tests. On this 1M one-shot
  workload it saved 3,464 MiB peak VRAM and changed total TTFT by only 0.256 s,
  because the approximately 671 s prefill dominates.
- For multi-turn workloads, use the isolated reselection numbers rather than
  the one-shot end-to-end percentage: CPU placement added about 0.86 s to
  scoring and 1.34 s to the measured semantic reselection in this run.
