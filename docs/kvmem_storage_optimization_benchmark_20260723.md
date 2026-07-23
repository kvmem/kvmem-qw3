# KVMem storage optimization benchmark (2026-07-23)

## Scope

This benchmark validates the consolidated storage profiles:

| Profile | Added mechanism | Target |
|---|---|---|
| `kvmem_init` | frozen exclusive-tier path | compatibility baseline |
| `opt_1` | heat-aware CPU admission/eviction | reduce SSD load volume |
| `opt_2` | inclusive SSD backing, bounded async writes, recycled pageable write slabs | reduce stage-out latency |
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

### Final semantic reselection

| Profile | SSD-in blocks | Stage-out | Stage-in wall | Assemble | Reselect total | TTFT |
|---|---:|---:|---:|---:|---:|---:|
| `kvmem_init` | 393 | 540.2 ms | 160.1 ms | 318.8 ms | 1066.6 ms | 54.465 s |
| `opt_1` | 446 | 593.2 ms | 164.2 ms | 323.2 ms | 1128.4 ms | 54.641 s |
| `opt_2` | 446 | 308.2 ms | 275.1 ms | 370.1 ms | 1001.2 ms | 53.590 s |
| `opt_3` | 446 | 225.6 ms | 84.4 ms | 274.3 ms | 632.2 ms | 53.204 s |

Against `kvmem_init`, cumulative `opt_3`:

- reduces the large-swap stage-out wall time by 58.2%;
- reduces stage-in wall time by 47.3%, despite loading 13.5% more SSD blocks;
- reduces the complete semantic reselection by 40.7%;
- reduces end-to-end TTFT by 1.26 seconds (2.3%). Long model prefill still
  dominates the roughly 53-second TTFT.

For the 446-block SSD input, `opt_3` issued 118 positional-I/O calls instead of
one call per block. The seven read batches spent 61.0 ms in I/O, but only
21.1 ms was exposed as wait; 39.9 ms (65.4%) overlapped H2D/other batch work.

The current tier file uses normal positional I/O, so recently written records
may also benefit from the Linux page cache. The separate direct-I/O probe above
is the physical-device reference; a future `O_DIRECT` backend is still needed
for strict page-cache-independent production measurements.

### Steady prefill-pressure stage-out

Mean of the last 20 pressure selections:

| Profile | Stage-out | Complete pressure selection |
|---|---:|---:|
| `kvmem_init` | 21.08 ms | 25.44 ms |
| `opt_1` | 27.03 ms | 31.51 ms |
| `opt_2` | 7.64 ms | 11.88 ms |
| `opt_3` | 4.86 ms | 9.17 ms |

The cumulative final profile reduces steady stage-out by 76.9%. SSD writes run
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

## Correctness and artifacts

- CTest: 12/12 passed.
- `NvmeKvTier` tests cover coalesced batch I/O and two simultaneous positional
  write batches, in addition to slot/LRU/read/write behavior.
- `git diff --check`: passed.

Primary raw logs and result summaries:

- `/tmp/qw3_opt_io_results/ioab_kvmem_init_cpu425_20260723_serve.log`
- `/tmp/qw3_opt_io_results/ioab_opt_1_cpu425_20260723_serve.log`
- `/tmp/qw3_opt_io_results/ioab_opt_2_final_cpu425_20260723_serve.log`
- `/tmp/qw3_opt_io_results/ioab_opt_3_final_cpu425_20260723_serve.log`
- `/tmp/qw3_opt_io_results/ioheat_kvmem_init_transcript70_mtp_20260723_serve.log`
- `/tmp/qw3_opt_io_results/ioheat_opt_1_transcript70_mtp_20260723_serve.log`

The matching manifests and per-sample JSONL/summary files use the same tag in
`/tmp/qw3_opt_io_results/`.
