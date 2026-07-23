# KVMem storage optimization benchmark (2026-07-23)

## Scope

This benchmark validates the consolidated storage profiles:

| Profile | Added mechanism | Target |
|---|---|---|
| `kvmem_init` | frozen exclusive-tier path | compatibility baseline |
| `opt_1` | heat-aware CPU admission/eviction | reduce SSD load volume |
| `opt_2` | inclusive SSD backing, bounded async writes, recycled pageable write slabs, writeback + page-cache release | reduce stage-out latency without duplicating the SSD arena in RAM |
| `opt_3` | coalesced reads, two recycled pinned read slabs, SSD-read/H2D pipeline | reduce stage-in latency |

The levels are cumulative. Retrieval scores, selected block IDs, and KV
contents are not intentionally changed.

## Hardware profile

The test filesystem is on a Samsung SSD 990 EVO Plus 1 TB. The device supports
PCIe Gen4 x4 but is currently linked at PCIe Gen3 x4.

Direct-I/O storage probe (`1 GiB`, `1 MiB` requests):

| Workload | Result |
|---|---:|
| initialization write | 2.81 GiB/s |
| sequential read | 2.97 GiB/s |
| random read, QD1 | 2.47 GiB/s |
| random read, QD8 | 3.47 GiB/s |
| random read, QD16 | 3.46 GiB/s |

QD8 is therefore the default bounded async-write queue depth. Both queue depth
and slab size remain runtime-tunable rather than device constants.

## Controlled one-sample I/O stress test

Common configuration:

- model: `Qwen3.6-27B-Q8_0.gguf`;
- KV dtype: fp16;
- LongMemEval-S sample index 0, 108,417 prompt tokens;
- context: 131,072;
- KVMem budget: 32,768;
- generation reserve: 1,024;
- block size: 32;
- sink blocks: 8;
- CPU budget: 4.25 GiB;
- NVMe arena: 8 GiB on `/tmp`;
- mean-k, query replay and immutable K enabled;
- MTP/thinking/judge disabled;
- one output token, so the result measures prefill/reselection rather than
  decode.

### Final semantic reselection (memory-bounded physical-I/O comparison)

All four rows below use the same memory-bounded policy. `opt_2` and `opt_3`
enable it by default; the compatibility `kvmem_init` and `opt_1` rows set
`QW3_KVMEM_DROP_PAGE_CACHE=1` explicitly. Completed write ranges are written
back and advised `POSIX_FADV_DONTNEED`; completed read ranges are likewise
released after their user-space buffer is populated. This prevents the kernel
page cache from becoming an unbudgeted second copy of the SSD tier.

| Profile | SSD-in blocks | Stage-out | Stage-in wall | Assemble | Reselect total | TTFT |
|---|---:|---:|---:|---:|---:|---:|
| `kvmem_init` | 393 | 1353.3 ms | 562.3 ms | 338.0 ms | 2301.2 ms | 57.030 s |
| `opt_1` | 446 | 1486.5 ms | 655.8 ms | 378.1 ms | 2568.3 ms | 57.602 s |
| `opt_2` | 446 | 451.1 ms | 651.7 ms | 335.7 ms | 1486.3 ms | 53.912 s |
| `opt_3` | 446 | 541.5 ms | 538.4 ms | 337.4 ms | 1465.1 ms | 53.806 s |

Against `kvmem_init`, cumulative `opt_3`:

- reduces the large-swap stage-out wall time by 60.0%;
- reduces stage-in wall time by 4.2%, despite loading 13.5% more SSD blocks;
- reduces the complete semantic reselection by 36.3%;
- reduces end-to-end TTFT by 3.22 seconds (5.7%). Long model prefill still
  dominates the roughly 54-second TTFT.

For the 446-block SSD input, `opt_3` issued 118 positional-I/O calls instead of
one call per block. The seven read batches spent 863.1 ms in aggregate worker
I/O; only 452.1 ms was exposed as wait, while 411.0 ms (47.6%) overlapped
H2D/other batch work. `opt_3` lowers stage-in by 17.4% relative to `opt_2`.

An earlier cached diagnostic measured a 632.2 ms total reselection, but it also
allowed the whole backing file to remain in Linux page cache. That number is
not a valid production comparison on a memory-bounded host and is superseded
by the table above.

### Steady prefill-pressure stage-out

Mean of the last 20 pressure selections:

| Profile | Stage-out | Complete pressure selection |
|---|---:|---:|
| `kvmem_init` | 53.12 ms | 57.40 ms |
| `opt_1` | 60.30 ms | 64.55 ms |
| `opt_2` | 7.89 ms | 12.15 ms |
| `opt_3` | 4.59 ms | 8.90 ms |

The cumulative final profile reduces steady stage-out by 91.4%. SSD writes run
behind a bounded queue; they are not silently discarded. Trace fields expose
submission time, buffer gather, backpressure, pending batches, completed bytes,
syscalls, and worker duration.

All four profiles produced the same first completion token (`You`). This is a
latency/copy-equivalence smoke test, not an answer-quality evaluation.

## `opt_1` multi-reselection test

A duplicate independent request is not a valid CPU-cache test because request
reset clears the block store and tier metadata. The valid test uses one
LongMemEval-S transcript request:

- sample index 70 (`question_id=0a995998`);
- 110,779 prompt tokens;
- 44 sessions / 171 transcript retrieval events;
- native MTP enabled because transcript replay requires it;
- otherwise the same 32K block-32 storage configuration.

Aggregate over all 171 semantic reselections:

| Profile | CPU-in blocks | SSD-in blocks | SSD-in volume | Stage-in total | Reselect total | TTFT |
|---|---:|---:|---:|---:|---:|---:|
| `kvmem_init` | 18,886 | 6,170 | 6.396 GiB | 6.000 s | 32.146 s | 110.023 s |
| `opt_1` | 22,341 | 2,690 | 2.788 GiB | 5.058 s | 31.429 s | 109.442 s |

`opt_1` reduces SSD-loaded blocks and bytes by 56.4%. The saved I/O lowers
aggregate stage-in by 15.7%; total reselection improves only 2.2% because
selection, KV assembly/re-RoPE, CPU H2D, and query replay remain outside this
optimization.

## Long-context host-memory regression

The first 2M-context, 224K-budget accuracy run completed two samples and then
lost the server during sample index 20 at roughly 545K prompt tokens. The
server log ended without a native exception. At that point the process could
hold close to the configured 72 GiB CPU tier while the 35 GiB buffered SSD file
was also eligible to remain in page cache. The failure signature and memory
accounting were consistent with host-memory pressure.

After enabling range writeback + page-cache release, the same sample was rerun
unchanged:

- prompt: 1,083,742 tokens;
- KVMem: 224K context + 32K generation, block 32, mean-k, query replay,
  immutable K, MTP-4;
- SSD file at completion: 34 GiB;
- observed system `Cached`: 26--29 GiB, dominated by the model mapping and not
  proportional to SSD-file growth;
- observed RSS near 946K prompt tokens: 82.1 GiB;
- observed `MemAvailable` at the same point: 59.9 GiB;
- result: completed normally, no truncation/error, official judge correct;
- TTFT / total latency: 661.6 / 673.9 seconds.

This is the stability gate that the cached implementation failed. The raw
artifacts are:

- `/data/chaidi/kvmem_eval/results/longmemeval_m_k224k_opt3_pagecachefix_s20_20260724_serve.log`
- `/data/chaidi/kvmem_eval/results/longmemeval_m_k224k_opt3_pagecachefix_s20_20260724_eval_20260723_161120.jsonl`

## Correctness and artifacts

- CTest: 12/12 passed.
- `NvmeKvTier` tests cover coalesced batch I/O and two simultaneous positional
  write batches, in addition to slot/LRU/read/write behavior.
- `git diff --check`: passed.

Primary raw logs and result summaries:

- `/tmp/qw3_opt_io_results/ioab_kvmem_init_nocache_cpu425_20260724_serve.log`
- `/tmp/qw3_opt_io_results/ioab_opt_1_nocache_cpu425_20260724_serve.log`
- `/tmp/qw3_opt_io_results/ioab_opt_2_pagecachefix_cpu425_20260724_serve.log`
- `/tmp/qw3_opt_io_results/ioab_opt_3_pagecachefix_cpu425_20260724_serve.log`
- `/tmp/qw3_opt_io_results/ioheat_kvmem_init_transcript70_mtp_20260723_serve.log`
- `/tmp/qw3_opt_io_results/ioheat_opt_1_transcript70_mtp_20260723_serve.log`

The matching manifests and per-sample JSONL/summary files use the same tag in
`/tmp/qw3_opt_io_results/`.
