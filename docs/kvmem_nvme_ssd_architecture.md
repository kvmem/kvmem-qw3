# KVMem PCIe NVMe SSD Architecture (Preliminary)

Status: superseded by the complete production design in
[`kvmem_ssd_complete_design.md`](kvmem_ssd_complete_design.md). This file
retains the original profiling notes and preliminary design history. The current
`NvmeKvTier` is a correctness-oriented synchronous file implementation and
does not implement this architecture yet.

## 1. Goals and assumptions

Target configuration:

- Qwen3.6-27B, FP16 KV;
- 10M logical context;
- 224K selected context plus 32K generation reserve;
- 32-token KVMem blocks;
- 96 GiB GPU and approximately 72 GiB KVMem CPU budget;
- PCIe NVMe SSD, not a rotating disk;
- immutable raw K, local-position MTP, and query replay enabled.

The portable first implementation stages through host memory. GPUDirect
Storage is an optional later backend, not a dependency of the base design.

### 1.1 Portability and runtime tuning

The values in this document are capacity examples, not compiled-in hardware
assumptions. Core KVMem code must not depend on a Samsung model, PCIe
generation, NUMA node, filesystem, queue depth, or a fixed slab/extent size.
Moving the same binary to another host must retain correctness and select a
reasonable transport automatically.

At server startup, a storage capability probe builds:

```cpp
struct StorageCapabilities {
    bool rotational;
    bool direct_io;
    bool io_uring;
    bool io_uring_fixed_buffers;
    bool gds;
    uint32_t logical_block_bytes;
    uint32_t physical_block_bytes;
    uint32_t max_request_bytes;
    uint32_t numa_node;
    double current_pcie_gts;
    double maximum_pcie_gts;
    uint32_t current_pcie_width;
    uint32_t maximum_pcie_width;
};
```

Unsupported or unavailable Linux sysfs fields are represented as unknown, not
as fatal errors. Device and GPU NUMA placement are discovered at runtime; code
must never assume node 0 or a particular PCI bus address.

The I/O backend is selected in this order:

```text
explicit override
-> supported calibrated backend
-> io_uring
-> preadv/pwritev worker pool
-> synchronous positional I/O correctness fallback
```

`O_DIRECT`, fixed buffers, and GDS are capability-gated. A filesystem or VM
that cannot support them therefore remains functional, with lower performance.

A short, optional calibration measures the KVMem record-size workload rather
than inferring performance from the product name or nominal PCIe generation:

- sequential read/write;
- random reads at the actual record and coalesced-extent sizes;
- queue-depth sweep;
- read-heavy mixed read/write latency;
- pageable and pinned host-to-device bandwidth.

The selected backend, queue depth, extent size, slab size/count, and I/O
concurrency are runtime `StorageTuning` values. Calibration results may be
cached using device identity, firmware, filesystem, kernel, GPU, and topology
as the key, and are invalidated when any of those change. Every value also has
a CLI/environment override so a deployment can enforce latency, memory, or
endurance policy without recompilation.

The current machine profile is useful only as a regression baseline. It must
not change record layout or correctness semantics, and a profile collected on
PCIe Gen3 must not be reused after moving the drive to Gen4/Gen5 hardware.

## 2. Capacity

For 16 standard-attention layers plus one local-position MTP layer:

```text
raw K bytes/token = 17 * 4 KV heads * 256 head_dim * 2 bytes
                  = 34,816 bytes

V bytes/token     = 34,816 bytes
```

At 10M tokens:

| Component | Approximate size |
|---|---:|
| immutable raw K | 324.25 GiB |
| all V | 324.25 GiB |
| raw K + V | 648.50 GiB |

The GPU active window and 72 GiB CPU cache reduce steady-state SSD residency,
but a fixed-capacity repository still needs space for roughly 600 GiB plus
metadata, alignment, temporary writeback, and evaluation artifacts. Production
deployment should reserve at least 1 TiB of free SSD capacity; a dedicated
2 TB or larger drive is preferable.

The current host has a Samsung NVMe device with about 676 GiB free. That is
enough only for a narrowly controlled 10M experiment and leaves insufficient
operational headroom for a production repository.

### 2.1 Current-host profile (regression baseline only)

`scripts/profile_kvmem_storage.py` discovers the storage device from a
filesystem path, reports sysfs capabilities, and runs an optional temporary
`O_DIRECT` benchmark. It does not contain a model table or assume a PCIe
generation:

```bash
python3 scripts/profile_kvmem_storage.py \
  --path /path/on/kvmem/ssd \
  --size-gib 8 \
  --duration 2
```

On 2026-07-23, the current host reported:

- Samsung SSD 990 EVO Plus 1TB, firmware `2B2QKXG7`;
- endpoint and direct upstream Root Port both capable of PCIe Gen4 x4;
- the negotiated link fixed at PCIe Gen3 x4, including during a 29 GB direct
  read;
- no correctable, nonfatal, or fatal AER counts at either endpoint;
- device and GPU both local to NUMA node 0;
- 128 KiB kernel maximum request size and 1023 request queue limit.

With a 2.125 MiB request (one fp16 Qwen3.6-27B raw-K+V KVMem block), the
temporary-file profile measured:

| Application concurrency | Read GiB/s | Mean latency | p99 latency |
|---:|---:|---:|---:|
| 1 | 2.74 | 0.76 ms | 0.85 ms |
| 4 | 3.38 | 2.45 ms | 2.54 ms |
| 8 | 3.43 | 4.83 ms | 4.92 ms |
| 16 | 3.44 | 9.62 ms | 10.32 ms |
| 32 | 3.34 | 19.79 ms | 23.29 ms |

Sequential read reached 3.28 GiB/s. This is consistent with a saturated Gen3
x4 path: increasing concurrency beyond 4--8 adds latency and does not recover
Gen4 bandwidth. These values are not defaults for another machine. The
calibration should run again after any drive, firmware, slot, kernel,
filesystem, or GPU-topology change.

The OS-visible evidence narrows the Gen3 cause to two classes:

1. firmware configures the Slot-4 Root Port target speed as Gen3; or
2. firmware requests Gen4, but link training falls back because of an adapter,
   backplane/cable, connector, or signal-integrity problem.

The unprivileged process cannot read PCIe Link Control 2 or the complete boot
log, so it cannot distinguish those two cases. A privileged read-only check is:

```bash
sudo lspci -s 0000:00:05.1 -vv | grep -E 'LnkCap|LnkSta|LnkCtl2'
sudo lspci -s 0000:01:00.0 -vv | grep -E 'LnkCap|LnkSta|LnkCtl2'
sudo journalctl -k -b | grep -Ei \
  'pcie|nvme|aer|downgrad|retrain|01:00.0|00:05.1'
```

`LnkCtl2` targeting 8 GT/s indicates a BIOS/firmware limit. A 16 GT/s target
with `LnkSta` at 8 GT/s indicates training fallback. Online register writes or
link retraining must not be attempted on this host because this NVMe is the
mounted root filesystem.

## 3. Why the current tier cannot use SSD performance

The current `NvmeKvTier`:

- owns one `FILE *`;
- performs one `fseek` plus `fread`/`fwrite` per block;
- calls `fflush` after every block write;
- allocates one `std::vector` for every selected SSD block;
- runs all reads serially inside one `std::async`;
- waits for all reads before starting SSD-buffer-to-GPU transfer;
- issues many 32 KiB CUDA copies for every block.

This serializes the request and keeps NVMe queue depth near one. It also makes
host allocation, libc locking, syscall setup, and CUDA launch/copy setup visible
on the critical path.

## 4. Persistent record layout

Use deterministic fixed-offset records indexed directly by `block_id`:

```text
record(block_id):
    raw_K[17 layers][32 tokens][4 heads][256 dims][fp16]
    V[17 layers][32 tokens][4 heads][256 dims][fp16]
    optional small validity/version metadata
```

The raw-K and V payloads are each approximately 1.0625 MiB; a complete record
is approximately 2.125 MiB and naturally much larger than the 4 KiB direct-I/O
alignment.

Advantages:

- `offset = data_base + block_id * aligned_record_bytes`;
- no hash lookup is needed for the common path;
- one selected block normally needs raw K and V together;
- one read can fetch both;
- consecutive block IDs form consecutive SSD extents;
- no compaction is required during a session.

Raw K becomes valid when target prefill captures the block. V becomes valid
when the block first stages out. Two validity bits and a generation number
prevent a partially written record from being consumed.

For experiments that need independent raw-K and V lifetimes, the same API can
use two deterministic files. The transfer scheduler should still combine their
I/O into one request batch.

## 5. NVMe I/O engine

Replace `FILE *` with an `NvmeIoEngine`:

```text
submit_read(batch)  -> IoTicket
submit_write(batch) -> IoTicket
poll_completions()
wait(ticket)
cancel(ticket)
```

Linux implementation:

- `open` with `O_DIRECT | O_CLOEXEC` after buffered-vs-direct A/B;
- `io_uring` fixed-file registration;
- fixed-buffer registration for the staging slabs;
- `preadv`/`pwritev` thread-pool fallback when `io_uring` is unavailable;
- queue depth chosen by calibration (the benchmark should cover at least
  8/16/32/64/128 where the device queue permits);
- separate logical read and write queues with read priority;
- no per-block `fflush` or `fsync`.

KVMem storage is an ephemeral inference cache. Durability after power loss is
not required, so per-block flushes provide no correctness benefit. A session
checkpoint can optionally issue one batched `fdatasync`.

Unlike an HDD design:

- do not read large unused gaps merely to avoid seeks;
- do not force one in-order request at a time;
- do not suppress useful read concurrency;
- submit independent 1–8 MiB extents concurrently until SSD bandwidth is
  saturated.

Adjacent selected records may still be coalesced to reduce submission overhead,
but only genuinely adjacent extents should be merged by default.

## 6. Registered slab pool

Allocate a bounded runtime-tuned host slab pool. Three or four 128–256 MiB slabs
are useful starting candidates, not fixed requirements:

1. allocate 4 KiB-aligned ordinary memory;
2. register it with CUDA using `cudaHostRegister`;
3. register the same ranges as `io_uring` fixed buffers;
4. reuse them for the lifetime of the server.

This gives one buffer that is simultaneously:

- valid for direct NVMe I/O;
- pinned for asynchronous PCIe H2D/D2H;
- bounded independently of the 72 GiB CPU cache.

Do not pin the long-lived 72 GiB cache. It remains pageable. The fixed slab
pool should normally remain below 1 GiB total.

Each slab has a state:

```text
FREE
READING
READY_HOST
H2D
SCATTER
DONE
```

An I/O completion moves `READING -> READY_HOST`. A CUDA event moves
`H2D/SCATTER -> DONE`, after which the slab can return to `FREE`.

## 7. SSD-to-GPU pipeline

The steady-state triple-buffer pipeline is:

```text
SSD read slab N+2
    || H2D slab N+1 to device staging
    || scatter/RoPE slab N into paged KV
    || independent model/query-replay computation
```

Do not issue one H2D per 32 KiB physical KV page. Each ready slab is copied to
one contiguous device staging buffer with one or a few large
`cudaMemcpyAsync` calls. A CUDA scatter kernel:

- locates each block payload inside the staging buffer;
- scatters V into the selected physical pages;
- scatters raw K and applies destination RoPE once;
- handles standard-attention and MTP layers;
- records a completion event for the slab.

Two device staging buffers are sufficient if scatter and H2D are serialized;
three allow fuller overlap when HBM bandwidth permits it.

## 8. CPU cache policy

At 10M, raw K cannot be permanently resident in 72 GiB DRAM. Raw K and V must
both be tierable.

Cache the pair at block granularity:

```text
GPU: selected working set
CPU: warm raw-K + V records
SSD: complete cold repository
```

Use a frequency/recency policy biased by retrieval score:

- mandatory sink and recent blocks;
- blocks selected in the last several turns;
- high mean-K score blocks;
- blocks adjacent to repeatedly selected evidence;
- dirty/pending-write blocks until SSD completion.

A paired raw-K/V cache avoids a raw-K hit plus V miss for the same selected
block. Partial records remain possible while a newly captured block has raw K
but has not yet staged out V.

CPU cache entries have:

```text
EMPTY | RAW_ONLY | COMPLETE
CLEAN | DIRTY | WRITING
```

A clean CPU entry may be discarded immediately. A dirty entry may be evicted
only after its asynchronous SSD write completes or after ownership transfers to
a writeback slab.

## 9. Reselection scheduling

Current order:

```text
score/select
-> synchronous stage-out
-> submit SSD reads
-> immediately wait
-> H2D
-> assemble
```

Target exact-selection order:

```text
score/select
-> submit SSD reads into slabs immediately
-> stage-out/free old GPU pages in parallel
-> consume completed slabs into GPU pages
-> assemble when required layers are ready
```

SSD reads do not need free GPU pages yet; they can fill host slabs while
stage-out frees pages. This reordering requires splitting page reservation from
I/O submission.

For a fast NVMe SSD, exact-selection prefetch is the first priority. A 224K
window contains 7168 blocks. In the worst all-cold case, raw K plus V is about
15.2 GiB. At 7 GiB/s it takes roughly 2.2 seconds; at 12 GiB/s roughly
1.3 seconds, before H2D/scatter. Those rates are examples only; scheduling and
latency estimates use the bandwidth measured for the active device. CPU-cache
hits lower this substantially.

Speculative top-M prefetch remains useful but should be conservative:

- mandatory sink/recent blocks can be submitted before scoring;
- retain last-turn selected blocks in GPU/CPU rather than rereading them;
- optionally prefetch 1.25–1.5x final K from partial-query scores;
- track wasted bytes and disable speculation when hit rate is poor.

SSD makes aggressive 2–4x overfetch unnecessary and increases write endurance
and PCIe contention without a proportional latency benefit.

## 10. Stage-out and writeback

Stage-out should be asynchronous:

```text
RESIDENT -> D2H -> WRITEBACK_HOST -> SSD_WRITING -> CLEAN_SSD
```

Rules:

- use large D2H gathers into write slabs;
- submit batched `io_uring` writes;
- never `fflush` per block;
- release a GPU page after D2H completion, not after SSD completion, provided
  the writeback slab owns the only dirty bytes;
- bound outstanding dirty writeback bytes;
- prioritize answer-producing reads over background writes;
- throttle writes when SSD thermal or latency telemetry crosses a threshold.

During long sequential prefill, block IDs increase monotonically, so raw-K and
V writes are naturally close to sequential even though reads are retrieval
driven.

## 11. Optional GPUDirect Storage

After the portable pipeline is stable, add a `cuFile` backend:

```text
NVMe -> GPU staging -> scatter/RoPE
```

It can remove host staging and one PCIe copy. It is optional because support
depends on the GPU, filesystem, kernel, driver, and storage topology. The
fixed-record and scatter design is shared with the host-staged backend, so GDS
does not require changing KVMem semantics.

Use an A/B benchmark; GDS is not automatically faster for small fragmented
requests, while 1–8 MiB records are favorable.

## 12. Interfaces to change

`NvmeKvTier` should become metadata plus an asynchronous engine:

```cpp
struct KvRecordLocation {
    uint64_t offset;
    uint32_t valid_parts;
    uint32_t generation;
};

struct IoExtent {
    uint32_t block_id;
    uint64_t file_offset;
    uint64_t bytes;
    uint32_t slab;
    uint64_t slab_offset;
};
```

`KvMemPrefetchState` should hold:

- an `IoTicket`, not a `std::future`;
- slab ownership and completion state;
- exact and speculative block sets;
- per-layer readiness events;
- bytes submitted/completed/wasted;
- CPU-cache hits and SSD misses.

The executor should expose:

```text
begin_reselect_io(plan)
progress_reselect_io()
finish_reselect_io()
```

This permits the scheduler and query replay to make progress without blocking
on all blocks at once.

## 13. Implementation order

1. Replace `FILE *` with file descriptors and deterministic fixed-offset
   records; remove per-block flushes.
2. Add a reusable capability probe and calibration tool covering
   buffered/direct I/O, record-derived request sizes, and queue depths supported
   by the active device. Store the result as runtime tuning, not constants.
3. Add `io_uring` asynchronous batches with fixed file and fixed host buffers.
4. Introduce the bounded registered slab pool.
5. Add bulk device staging plus V/raw-K scatter kernels.
6. Reorder reselection so exact SSD reads start before stage-out.
7. Make raw-K chunks SSD-tierable and change the CPU tier into a paired
   raw-K/V cache.
8. Add asynchronous writeback and clean-backing-copy tracking.
9. Add conservative top-M speculative prefetch.
10. Evaluate GPUDirect Storage.

## 14. Required telemetry and acceptance criteria

Record per reselection:

- CPU cache hit blocks/bytes;
- SSD read/write blocks, extents, bytes, and queue depth;
- submit-to-first-completion and submit-to-last-completion;
- effective SSD read/write GiB/s;
- pinned-slab occupancy and starvation;
- H2D/D2H GiB/s;
- scatter/RoPE time;
- SSD time hidden by stage-out/query replay;
- speculative hit rate and wasted bytes;
- visible reselect wait;
- SSD temperature/throttling if available.

Initial targets on a suitable PCIe NVMe SSD:

- no per-block allocation, `fflush`, or serialized `fread`;
- at least 70% of the target SSD's measured large-random-read bandwidth;
- at least 15 GiB/s pinned H2D on a capable host/GPU path;
- less than 1 GiB permanently pinned host staging;
- greater than 70% overlap of SSD read time with stage-out/H2D/scatter where
  dependency permits;
- no violation of the configured CPU/GPU budgets;
- byte-identical answers and selected-block contents versus the synchronous
  reference path.
