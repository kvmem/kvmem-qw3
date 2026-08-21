# KVMem SSD Complete Design

Status: proposed production architecture. This document defines the target
semantics, ownership, layout, scheduling, portability, rollout, and acceptance
criteria. It does not describe the current synchronous `NvmeKvTier` as if that
implementation were complete.

## 1. Decisions

The production SSD path uses these decisions:

1. GPU contains one active, position-baked K/V working set only. There is no
   second working-K mirror.
2. Unrotated raw K and V are written through to SSD while prefill is running.
   SSD is an inclusive clean backing store for every completed historical
   block, not merely an overflow location reached during reselection.
3. CPU DRAM is a bounded, pageable read cache for complete raw-K/V records.
   Long-lived cache memory is not pinned.
4. Small bounded pinned slabs are shared by direct SSD I/O and CUDA DMA.
5. SSD records are split into calibrated layer stripes. On the current machine,
   four attention layers per 512 KiB stripe preserves nearly all measured SSD
   bandwidth while allowing layer-group I/O to overlap query replay.
6. Mean-K remains exact. At 10M it is stored in pageable CPU chunks and streamed
   through bounded GPU tiles unless spare VRAM permits a resident index.
7. `io_uring` is the preferred asynchronous backend. A positional-I/O worker
   pool and a synchronous reference backend preserve portability.
8. Storage and I/O resources are server-global. Each executor gets an isolated
   session namespace; executors must not open and truncate the same filename.
9. Query replay and immutable raw K remain enabled. A cold K is baked from raw
   K directly into its selected window position, and MTP uses the same local
   position frame.
10. Hardware-dependent values are runtime tuning results, never model/device
    constants in core logic.

## 2. Target and exact capacity

Primary target:

- Qwen3.6-27B;
- fp16 KV;
- 16 standard-attention layers plus one local-position MTP layer;
- four KV heads, head dimension 256;
- 32-token KVMem blocks;
- 10M logical tokens;
- 224 Ki tokens selected context plus 32 Ki generation reserve;
- 96 GiB physical GPU, with deployments optionally imposing a lower process
  budget;
- 72 GiB total KVMem host-memory budget;
- dedicated PCIe NVMe SSD for production.

Per token:

```text
raw K = 17 * 4 * 256 * 2 bytes = 34,816 bytes = 34 KiB
V     = 17 * 4 * 256 * 2 bytes = 34,816 bytes = 34 KiB
total = 68 KiB/token
```

Per 32-token block:

```text
raw K = 1.0625 MiB
V     = 1.0625 MiB
record = 2.125 MiB
```

At 10M tokens:

| Component | Size |
|---|---:|
| raw K | 324.25 GiB |
| V | 324.25 GiB |
| inclusive SSD K+V repository | 648.50 GiB |
| 16-layer fp16 mean-K index | 9.54 GiB |
| per-block metadata at 128 bytes/block | 38.1 MiB |

An inclusive SSD repository deliberately does not subtract the GPU and CPU
cache from SSD capacity. This costs more SSD space but removes reselection-time
writeback from the normal critical path.

A production filesystem should have at least:

```text
repository capacity
+ 5% allocation/alignment reserve
+ 32 GiB operational free-space reserve
```

For 10M fp16 this means more than 713 GiB free before evaluation artifacts.
At least 1 TiB free is acceptable; a dedicated 2 TB or larger device is
preferred. The current system SSD's roughly 676 GiB free space is enough for
smaller validation runs, but not a safe full 10M production allocation.

The 72 GiB host budget is divided at runtime, not statically:

```text
mean-K exact index       about 9.54 GiB at 10M
metadata                 less than 0.1 GiB
pinned read/write slabs  normally 0.25--1.0 GiB
pageable raw-K/V cache   all remaining budget
```

With approximately 62 GiB left for full records, the CPU data cache covers
roughly 956K tokens.

## 3. End-to-end architecture

```text
                                  +----------------------+
query -> exact mean-K tiles ----> | Retrieval / top-k    |
                                  +----------+-----------+
                                             |
                                      selected block IDs
                                             |
                           +-----------------v------------------+
                           | SelectionTransaction / planner     |
                           | retain, load, release, remap        |
                           +------+--------------------+---------+
                                  |                    |
                         CPU-cache hits          SSD-clean misses
                                  |                    |
                                  +---------+----------+
                                            |
                                  registered read slabs
                                            |
                                  bulk H2D staging buffers
                                            |
                            scatter V + raw-K->RoPE(K)
                                            |
                                  bounded GPU page pool
                                            |
                             query replay -> MTP -> decode

prefill raw K + V -> GPU pack -> pinned write slab -> async SSD write
                               \-> optional CPU cache admission
```

Server-global components:

```text
KvStorageManager
  - StorageCapabilities
  - StorageTuning
  - KvRecordAllocator
  - KvIoBackend
  - ReadSlabPool / WriteSlabPool
  - global capacity admission and I/O scheduling
```

Per-executor components:

```text
KvStorageSession
  - block_id -> record_slot/generation mapping
  - block residency and backing metadata
  - pending I/O tickets
  - CPU cache policy state
  - MeanKIndexStore namespace
```

`KvMemStore` remains responsible for block chronology, scores, mandatory
blocks, and deterministic selected-window order. It must not perform file I/O.

## 4. Correctness and position invariants

Storage optimization must not alter retrieval or model semantics.

1. Raw K is captured before RoPE and is position independent.
2. V is stored byte-for-byte and is never position transformed.
3. Mean-K is computed in the same content frame as raw K.
4. Selected blocks are ordered by original block ID and packed from local
   window position zero.
5. A cold K is read from CPU/SSD and receives destination-window RoPE exactly
   once during GPU scatter.
6. A retained GPU K may use a bounded in-place delta re-RoPE. The existing
   remap-count and cumulative-angle thresholds force a raw refresh.
7. No K used by normal attention, MTP, prefill replay, or decode may be baked
   beyond the model's context limit. With 224K+32K, every active position stays
   below 256K.
8. MTP raw K follows the same unrotated storage and local-window materialization
   as standard attention.
9. Query replay starts from the valid checkpoint boundary only after the
   selected context required by its current layer is ready.
10. I/O completion never commits `baked_pos`, residency, or page ownership.
    Those metadata changes occur only when the complete selection transaction
    commits successfully.

The storage path therefore fixes the two unsafe patterns:

- persisting already-RoPE'd K at million-token absolute positions;
- repeatedly treating a lossy working K as the immutable authority.

## 5. SSD layout

### 5.1 Layout descriptor

Every storage arena has an immutable descriptor:

```cpp
struct KvStorageLayout {
    uint64_t model_fingerprint;
    uint32_t block_tokens;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    KvDType dtype;
    std::vector<uint32_t> attention_layer_ids;
    uint32_t layers_per_stripe;
    std::vector<uint64_t> stripe_record_bytes;
    uint64_t full_record_bytes;
    uint32_t direct_io_alignment;
};
```

A backend instance serves only matching layouts. Different model, dtype, block
size, or stripe grouping gets a separate arena.

### 5.2 Layer stripes

One attention layer contributes 64 KiB raw K plus 64 KiB V for a 32-token fp16
block. With four layers per stripe:

```text
stripe 0: standard layers 0..3     512 KiB/block
stripe 1: standard layers 4..7     512 KiB/block
stripe 2: standard layers 8..11    512 KiB/block
stripe 3: standard layers 12..15   512 KiB/block
stripe 4: local-position MTP       128 KiB/block
total                               2.125 MiB/block
```

Each stripe has a global file:

```text
stripe_00.bin
stripe_01.bin
...
```

Within a stripe:

```text
offset = aligned_header_bytes + record_slot * stripe_record_bytes
payload = raw_K[group_layers] || V[group_layers]
```

The shared `record_slot` identifies the same block in every stripe. Separate
stripe files avoid max-context-sized holes and allow the scheduler to request
the layer group needed next.

The current SSD profile measured:

| Request | Useful read bandwidth |
|---|---:|
| 128 KiB, application concurrency 32 | 2.63 GiB/s |
| 512 KiB, application concurrency 16 | 3.36 GiB/s |
| 2.125 MiB, application concurrency 8 | 3.43 GiB/s |

Four-layer stripes retain almost all large-record bandwidth while enabling
query-replay overlap. This is a profile result, not a compiled default.
Calibration may choose 1, 2, 4, 8, or all layers per stripe on another host.

### 5.3 Global record allocation and generations

The backing file is owned once by `KvStorageManager`. A server-global slot
allocator prevents capacity overcommit and filename collisions.

Each session maintains:

```cpp
struct BlockRecordRef {
    uint64_t record_slot;
    uint64_t generation;
};
```

Record slots are normally allocated sequentially, preserving temporal locality.
When query replay truncates and rebuilds a suffix, it allocates new slots for
the new generation. An old in-flight write therefore lands in an unreachable
old slot rather than racing a newer block at the same offset. The old slot is
returned only after its I/O reference count reaches zero.

For a production fixed-capacity arena, stripe files are preallocated with
`fallocate`. This fails at server admission instead of producing `ENOSPC` after
hours of prefill. Admission also reserves transient copy-on-write slots for
replayed suffixes and in-flight generations; the 10M/700-GiB recommendation
leaves roughly 51 GiB inside the arena for this purpose.

### 5.4 Metadata

The cache is ephemeral. Authoritative residency metadata remains in memory:

- owner session and block ID;
- generation;
- per-stripe validity and clean masks;
- CPU cache slot;
- GPU main/MTP pages;
- in-flight read/write references;
- remap counters and baked position.

No small header is placed before every record, which would complicate
`O_DIRECT` and add write amplification. Optional recoverable sessions require
a separate versioned metadata journal and are outside the first implementation.

## 6. Residency and ownership

The current exclusive `KvTier` enum is insufficient: a hot block may
simultaneously have an active GPU copy, a CPU cache copy, and clean SSD backing.
Replace it with explicit residency and backing state.

Per stripe:

```text
data authority:
  GPU_DIRTY
  WRITE_SLAB
  SSD_CLEAN
  ERROR

inclusive copies:
  GPU_PRESENT
  CPU_PRESENT
  SSD_VALID
```

Normal lifecycle:

```text
prefill completes block
-> GPU_DIRTY
-> D2H pack completes: WRITE_SLAB owns bytes
-> GPU may be released if necessary
-> SSD write completes: SSD_CLEAN
-> optional CPU cache copy remains inclusive
```

Stage-in:

```text
GPU_PRESENT                         -> retain
CPU_PRESENT                         -> CPU gather -> H2D
WRITE_SLAB                          -> wait D2H, then H2D from slab
SSD_CLEAN                           -> SSD read -> H2D
none of the above                   -> correctness error
```

Rules:

- never release the last valid owner;
- never reuse a GPU page before its final reader and D2H event complete;
- never reuse a slab while SSD or CUDA still references it;
- never mark SSD clean until its write CQE succeeds;
- never silently substitute zero or stale KV after an I/O error.

## 7. Prefill write-through

Write-through is the most important latency change.

With immutable source K and local-position MTP, tier records contain the 16
normal-attention V pages plus one MTP V page. Raw K remains in its independent
position-free source store and is not duplicated in the SSD record. For the
current Qwen3.6-27B configuration:

```text
2048 tokens / 32 = 64 complete blocks
1 block = 17 * 64 KiB = 1.0625 MiB
1 chunk = 64 * 1.0625 MiB = 68 MiB
```

The implemented `opt_2`/`opt_3` SSD pipeline is:

```text
MAIN + MTP compute chunk N
  || packed gather + D2H for chunk N-1
  || SSD write for chunk N-2
```

At the MAIN+MTP completion boundary for each chunk:

1. find newly completed full blocks;
2. GPU-gather their paged V into one contiguous device staging range;
3. enqueue one D2H per slab on the dedicated KV copy stream;
4. retain the original GPU pages, so later prefill is unaffected;
5. at the next chunk boundary, wait only for any uncovered D2H tail, reserve
   any heat-aware CPU-cache destinations, then hand the pinned slab to a
   bounded background worker;
6. in that worker, populate admitted CPU-cache records and issue the positional
   SSD writes while the next GPU prefill chunk runs;
7. mark the block `in_flight` only after D2H owns a complete copy;
8. allow pressure selection to release the GPU page immediately when either an
   in-flight slab or clean SSD copy exists;
9. publish CPU-copy metadata and mark SSD clean only after the worker
   completes.

Four 64 MiB pinned slabs are the portable default. A 2048-token chunk occupies
two slabs (68 MiB), so four slabs form two chunk-sized sets: one may remain
owned by SSD writes while the next receives D2H. Override the bounded pool with
`QW3_KVMEM_WRITEBACK_SLABS` (2 through 16) and the common slab size with
`QW3_KVMEM_IO_SLAB_MIB`.

This path is enabled by default only when all of the following are true:

- optimization level is `opt_2` or `opt_3`;
- an SSD tier exists;
- immutable source K is enabled;
- MTP is absent or uses local-position K.

Set `--kvmem-opt-stage-out off` for a matched compatibility baseline. The old
`QW3_KVMEM_PREFILL_WRITEBACK` and `QW3_KVMEM_PREFILL_CPU_ADMIT` experimental
overrides have been removed; CPU admission is part of the unified stage-out
policy so proactive write-through does not turn later CPU hits into SSD reads.
Legacy mutable-K or baked-position MTP configurations continue through the
selection-time stage-out path because a non-destructive write-through copy
cannot canonicalize their working K in place.

In `opt_3`, CPU hits and SSD misses remain separate producers but may appear in
the same selection. CPU records use packed gather/H2D while SSD records use
coalesced reads. The mixed configuration preallocates two 64 MiB pinned read
slabs for each producer, avoiding dynamic pinned allocation when one pipeline
temporarily consumes the other one's buffers.

At 3K prefill tokens/s, KV production is:

```text
3000 * 34 KiB = 99.6 MiB/s
```

The current SSD sustained more than 1.5 GiB/s even in the conservative
temporary-file initialization. Normal prefill therefore uses only a small
fraction of write bandwidth and should hide nearly all persistence time.

If SSD writeback falls behind:

- keep a bounded `pending_write_bytes`;
- apply backpressure only when write slabs or GPU headroom approach exhaustion;
- prioritize demand reads over background writes;
- emit a warning when the moving write rate is below the KV production rate.

Partial tail blocks stay GPU-resident. They are packed when full or when an
explicit session checkpoint requests a drain.

With `QW3_KVMEM_PERF_TRACE=1`, `[kvmem-writeback]` records show:

- `event=d2h_submit`: completed position, block count, and bytes;
- `event=ssd_submit`: exposed D2H wait and the elapsed overlap window;
- `event=ssd_complete`: device write duration, queue depth, and whether the
  batch came from prefill write-through or pressure fallback.

## 8. Exact mean-K index at 10M

The current all-layer `g_kbar_multi_` is GPU resident. At 10M/block-32/fp16 it
requires about 9.54 GiB, which is incompatible with a strict 48 GiB total GPU
target once weights, active KV, scratch, and staging are included.

Introduce `MeanKIndexStore` with runtime placement:

```text
GPU_RESIDENT     when VRAM budget permits
CPU_STREAMED     production default for 10M
SSD_STREAMED     capacity fallback, not a latency target
```

CPU layout is chunked:

```text
[chunk][standard_layer][block_in_chunk][kv_head][head_dim]
```

Suggested starting chunk: 2048 blocks, approximately 64 MiB fp16. Two pinned
GPU-transfer buffers permit:

```text
pass 1:
  CPU copy tile N+1
    || H2D tile N
    || compute local max/sum tile N-1

combine all tile statistics into the exact global softmax max/denominator

pass 2:
  CPU copy tile N+1
    || H2D tile N
    || recompute dot, normalize, and accumulate block scores for tile N-1

global top-k over the final block scores
```

Two passes are required because the current retrieval score is a softmax over
all pages independently for each query/head/layer, followed by aggregation.
Normalizing each tile independently would change the ranking. Retaining every
query/head/layer/block logit is impractical, so the exact bounded-memory
algorithm recomputes dot products after the global denominators are known,
matching the existing over-8192-block two-pass logic.

The full CPU index remains pageable. Only bounded transfer tiles are pinned.
The final score output for 312,500 blocks is about 1.2 MiB fp32, so either GPU
top-k merge or one final score D2H is reasonable.

Primary results must use the exact fp16 index. FP8 index storage is an explicit
accuracy A/B, not an automatic memory fallback.

If neither GPU nor CPU can hold the exact index, configuration logs the
estimated SSD scan latency and requires an explicit opt-in. Scanning the entire
index from SSD on every turn would otherwise hide a severe latency regression.

## 9. Reselection transaction

### 9.1 Planning

Scoring produces selected block IDs without mutating live residency.

`SelectionTransaction` computes:

- retained GPU blocks;
- incoming CPU hits;
- incoming pending-write hits;
- incoming SSD reads;
- retained blocks requiring raw-K refresh;
- clean outgoing GPU blocks;
- dirty outgoing blocks;
- destination window positions and page reservations.

The transaction keeps the old window valid until commit.

### 9.2 Exact demand pipeline

```text
1. submit known mandatory sink/recent reads
2. exact mean-K scoring
3. submit exact selected SSD reads immediately
4. gather CPU hits in parallel
5. release clean outgoing pages after their last-reader event
6. transfer ready stripe slabs H2D
7. scatter V and apply destination RoPE to raw K
8. publish per-stripe/layer readiness events
9. query replay waits only for its next required layer group
10. finish MTP stripe, commit transaction, decode
```

SSD reads start before stage-out finishes because host slabs do not require free
GPU pages. Stage-out is normally only page release: write-through already
created clean SSD backing.

### 9.3 Stripe/query-replay overlap

With four-layer stripes:

```text
load/scatter stripe 0
replay layers in stripe 0 || load/scatter stripe 1
replay intervening layers || load/scatter stripe 2
...
load MTP stripe before first draft step
```

The executor adds `wait_kv_ready(layer)` before a standard-attention layer.
DeltaNet, norm, projection, and FFN compute from earlier layers can hide later
stripe reads.

If query replay is disabled for an ablation, all selected stripes must be ready
before decode. Storage correctness remains unchanged.

### 9.4 Retained K

Retained selected pages avoid SSD/CPU transfer. Their K handling is:

- no position change: no work;
- small bounded move: in-place delta re-RoPE;
- cold block, remap threshold, cumulative-angle threshold, or out-of-range
  baked position: refresh raw K from CPU/SSD and bake once.

V never needs remapping.

### 9.5 Speculation

Speculative I/O is optional and lower priority:

- sink and recent are safe to prefetch before scoring;
- last-turn blocks should be retained rather than reread;
- partial-query top-M may prefetch at most 1.25--1.5x final K;
- wasted bytes and hit rate automatically disable unhelpful speculation.

Exact selected reads always preempt speculative reads and background writes.

### 9.6 Horizontal scheduling timelines

The diagrams below are conceptual and not to scale. A box spanning the same
horizontal interval as a box on another lane is eligible to overlap, subject
to the CUDA/I/O events described below.

#### Steady-state prefill

```text
time -----> |       P0       |       P1       |       P2       |       P3       |

GPU compute | prefill C0      | prefill C1      | prefill C2      | prefill C3      |
            |                 |                 |                 |                 |
GPU storage | capture raw C0  | pack C0,D2H W0 | pack C1,D2H W1 | pack C2,D2H W2 |
 / DMA      |                 |                 |                 |                 |
CPU storage | free write slab | own W0/cache C0 | own W1/cache C1 | own W2/cache C2 |
            |                 | submit C0       | submit C1       | submit C2       |
SSD         | idle/old writes | write C0        | write C1        | write C2        |
```

`pack` is a GPU kernel and may contend with prefill for SM/HBM, so it is short,
event-driven work rather than assumed perfect overlap. Its following D2H uses a
copy engine and can overlap more fully with compute. SSD write C0 overlaps
prefill C2 and later. Once W0's D2H completes, the write slab is a valid owner;
GPU pages no longer have to wait for the SSD write itself before release.

#### Exact reselection and query replay

`S0..S3` are four-layer standard-attention stripes and `SM` is the MTP stripe.

```text
time -----> | query ready | mean-K pass 1 | mean-K pass 2 | load S0 / plan |
            |             |               | + exact top-k |                |

GPU compute | finish query| dot + max/sum | dot+normalize | wait/short plan|
            | capture     |               | + top-k       |                |
GPU storage | old window  | index H2D A/B | index H2D A/B | retain pages  |
 / DMA      | remains live|               |               | release clean |
            |             |               |               | H2D/scatter S0|
CPU storage | query state | index tiles   | index tiles   | gather S0 hits |
            |             | A/B           | A/B           | SSD S0 slabs  |
SSD         | low-priority| mandatory only| mandatory only| exact read S0 |
            | writeback   | prefetch      | prefetch      |                |

time -----> | replay G0    | replay G1    | replay G2    | replay G3    |
            | + load S1    | + load S2    | + load S3    | + load SM    |

GPU compute | replay layers| replay layers| replay layers| replay layers|
            | in group 0   | in group 1   | in group 2   | in group 3   |
GPU storage | H2D/scatter S1|H2D/scatter S2|H2D/scatter S3|H2D/scatter SM|
 / DMA      | while G0 runs| while G1 runs| while G2 runs| while G3 runs|
CPU storage | gather/handoff|gather/handoff|gather/handoff|gather/handoff|
            | S1 slabs     | S2 slabs     | S3 slabs     | SM slabs     |
SSD         | read S1      | read S2      | read S3      | read SM      |

time -----> | transaction commit | MTP draft/verify + normal decode ------------>

GPU compute | final replay tail   | decode                                          |
GPU storage | selected KV active  | append generated K/V in local window            |
CPU storage | recycle read slabs  | cache/write complete generated blocks           |
SSD         | background writes   | low-priority write-through                      |
```

The exact selected set is unknown until mean-K pass 2 finishes. Before that,
only mandatory sink/recent blocks and conservative speculation may be read.
The old committed window remains valid throughout scoring. After top-k, exact
reads are submitted before outgoing page release completes.

Replay group G0 starts only when every selected block's S0 data is present on
GPU. It does not wait for S1..SM. Before each later attention layer, the compute
stream waits on that stripe's readiness event. If storage is slower than replay,
the compute lane shows a bubble at that boundary; if replay is slower, later
stripes finish early and wait without blocking compute.

#### Inside one stripe load

For one selected stripe, records are packed into several fixed read slabs:

```text
time -----> |      Q0       |      Q1       |      Q2       |      Q3       |

SSD         | read slab R0  | read slab R1  | read slab R2  | read slab R3  |
CPU storage | R0 READING    | R0 READY      | R1 READY      | R2 READY      |
            | fill cache hits| fill R1/cache | fill R2/cache | fill R3/cache |
GPU DMA     | idle          | H2D R0->D0    | H2D R1->D1    | H2D R2->D0    |
GPU kernel  | idle          |               | scatter D0    | scatter D1    |
            |               |               | + raw-K RoPE  | + raw-K RoPE  |
```

Dependencies:

```text
SSD CQE(Rn) -> Rn READY
Rn READY -> H2D(Rn)
H2D event(Dn) -> scatter/RoPE(Dn)
scatter event(Dn) -> slab/device-buffer reuse
all stripe entries scattered -> kv_ready(stripe)
kv_ready(stripe) -> replay may enter that layer group
```

CPU cache hits are gathered into unused ranges of the same read slabs while
SSD fills other slabs. Separate read and write pools prevent prefill writeback
from consuming buffers required by exact reads.

#### What overlaps and what does not

Eligible overlap:

- prefill compute with previous-chunk D2H and SSD writes;
- mean-K tile CPU copy, H2D, and GPU scoring as a double-buffer pipeline;
- SSD reads with CPU-cache gather and clean GPU-page release;
- SSD read of stripe N+1 with H2D/scatter/replay of stripe N;
- low-priority write-through with compute when demand reads are not starved.

Hard dependencies:

- exact non-mandatory SSD reads wait for final top-k IDs;
- a slab H2D waits for its SSD/CPU fill completion;
- scatter waits for H2D;
- an attention layer waits until all selected K/V for its stripe are scattered;
- a dirty GPU page cannot be released until D2H has transferred ownership;
- a page/slab/device buffer cannot be reused before its final event;
- transaction commit waits for every stripe needed by final replay/MTP.

Resource contention means eligible overlap is not automatically free:

- pack/scatter and Transformer compute both use GPU SM/HBM;
- SSD DMA, CPU memory copies, and GPU DMA share host-memory/I/O paths;
- concurrent SSD writes may increase demand-read tail latency.

The runtime calibration therefore measures concurrent, not only isolated,
throughput and may serialize a contended kernel even though its dependency graph
allows overlap.

## 10. Direct-I/O engine and slabs

### 10.1 Backend interface

```cpp
class KvIoBackend {
public:
    virtual IoTicket submit_reads(std::span<const IoExtent>) = 0;
    virtual IoTicket submit_writes(std::span<const IoExtent>) = 0;
    virtual size_t poll(std::span<IoCompletion>) = 0;
    virtual void cancel(SessionId, uint64_t generation) = 0;
};
```

Backends:

1. `IoUringKvIoBackend`: production Linux path, registered files and buffers.
2. `ThreadedKvIoBackend`: `preadv/pwritev` worker-pool fallback.
3. `SyncKvIoBackend`: deterministic correctness and fault-injection reference.
4. optional `CuFileKvIoBackend`: later GDS path.

No backend uses `FILE *`, shared seek position, per-block allocation,
per-block `fflush`, or one `std::async` loop.

### 10.2 Direct I/O

Use `O_DIRECT | O_CLOEXEC` when the filesystem accepts it. All stripe offsets,
lengths, and slab addresses satisfy the discovered alignment. Buffered I/O is a
functional fallback and must be reported prominently because page-cache
results have different memory accounting.

KVMem inference storage is ephemeral. Normal block writes do not call `fsync`
or `fdatasync`. An optional durable session checkpoint may issue one batched
sync after all prior writes complete.

### 10.3 Slab pools

Read and write slabs are separate so writeback cannot consume every buffer
needed by a demand read.

Suggested search space:

```text
slab bytes:   32, 64, 128, 256 MiB
read slabs:   2..4
write slabs:  2..3
device slabs: 2..3
```

Host slabs:

1. page-aligned allocation;
2. `cudaHostRegister`;
3. `io_uring` fixed-buffer registration;
4. NUMA placement local to the SSD/GPU path;
5. counted inside the configured host-memory budget.

Read slab state:

```text
FREE -> FILLING_CPU/READING_SSD -> READY
     -> H2D -> SCATTER -> FREE
```

Write slab state:

```text
FREE -> D2H -> READY -> SSD_WRITING -> FREE
```

Every transition is driven by an I/O completion or CUDA event, never by an
assumed delay.

### 10.4 Bulk GPU transfers

One slab uses one or a few `cudaMemcpyAsync` operations into a contiguous
device staging buffer. A metadata array maps packed entries to:

- block and generation;
- stripe and layer range;
- source offset;
- destination physical page;
- destination window position;
- valid token count.

A fused kernel scatters V and applies RoPE while scattering raw K. This replaces
thousands of 32 KiB copies and separate per-block remap launches.

## 11. CPU cache

The CPU tier is an inclusive cache, not the sole owner of evicted data.

Data storage:

- pageable, fixed-size full-record slots;
- chunked `mmap`/arena allocation rather than one `new[]` per block;
- no permanent pinning;
- exact accounting inside `--kvmem-cpu-gb`.

Initial policy: segmented LRU/2Q.

Protected entries:

- sink blocks;
- current and previous selected windows;
- blocks selected repeatedly;
- high-score evidence;
- dirty or in-flight records.

Probationary entries:

- one-off SSD reads;
- adjacent speculative records;
- newly written history not yet reused.

Admission uses reuse frequency plus retrieval score. A CPU cache hit is copied
into a pinned read slab in large batches; direct H2D from pageable per-block
buffers is forbidden.

CPU cache copies may remain while the block is GPU resident. This intentional
duplication improves repeated-turn reselection and is bounded by the cache
budget.

## 12. Multi-session scheduling

The current per-executor `NvmeKvTier` opens
`qw3_kvmem_nvme.bin` with `w+b`; multiple executors in one directory can
truncate each other's backing. The new manager eliminates this design.

Global admission:

- reserve record slots before accepting a session's declared maximum;
- support a configurable overcommit policy only when explicitly enabled;
- include mean-K, metadata, slabs, and CPU cache in host admission;
- include active KV and device staging in GPU admission.

I/O priorities:

```text
P0 exact selected reads
P1 mandatory/query-replay reads
P2 dirty ownership writes needed to free GPU/slabs
P3 normal prefill write-through
P4 speculative reads
```

Fairness:

- per-session outstanding-byte credits;
- global queue-depth cap;
- no session may occupy every read slab;
- cancellation is generation based;
- session destruction waits only for ownership-safe cleanup, not optional
  durability.

## 13. Runtime capability discovery and calibration

No Samsung model, PCIe generation, NUMA node, queue depth, slab size, or layer
stripe is hard-coded.

Probe:

- rotational flag;
- logical/physical block sizes;
- maximum request size and queue limit;
- filesystem direct-I/O support;
- io_uring feature support;
- SSD PCI address, NUMA node, link width/speed;
- GPU PCI address and NUMA topology;
- optional GDS support.

Calibrate:

- direct versus buffered I/O;
- stripe request sizes derived from the actual layout;
- read concurrency and mixed read/write latency;
- slab sizes/counts;
- pageable CPU gather bandwidth;
- pinned H2D/D2H bandwidth;
- scatter/RoPE throughput;
- concurrent SSD-read + H2D interference.

Cache the result using:

```text
device identity + firmware + filesystem + kernel
+ model/layout + GPU + PCIe/NUMA topology
```

Invalidate it whenever a key field changes.

Runtime selection order:

```text
explicit CLI override
-> valid cached calibration
-> short startup calibration
-> conservative capability-derived default
```

The reusable probe is:

```bash
python3 scripts/profile_kvmem_storage.py --path /ssd/path
```

The current host would begin with four-layer/512 KiB stripes and application
read concurrency around 16. A Gen4/Gen5 system recalibrates independently.

## 14. Failure handling

### Read error

- use a valid GPU/CPU/write-slab copy if available;
- mark the SSD slot degraded and schedule rewrite;
- if no valid copy exists, fail the request clearly;
- never return fabricated KV;
- optional token replay/recomputation is a slow, explicit recovery mode, not a
  silent fallback.

### Write error

- retain ownership in GPU, CPU, or write slab;
- stop releasing the last owner;
- retry within a bounded policy;
- degrade to CPU-only only if capacity admission proves it safe;
- otherwise fail before corrupting model state.

### Capacity error

- preallocate/admit before long prefill;
- treat unexpected `ENOSPC` as backend degradation;
- do not continue after dropping KV.

### io_uring/direct-I/O unavailable

- fall back to the worker-pool backend, then synchronous positional I/O;
- keep the same record format and state machine;
- log the backend and expected performance difference.

### Transaction failure

- retain the previous committed window;
- cancel or ignore new-generation I/O;
- release uncommitted page reservations and slabs;
- commit block residency and baked positions only after all required events
  succeed.

### Restart

The first implementation treats the SSD arena as ephemeral and reinitializes
it. Recoverable sessions need a checksummed metadata journal and model/layout
compatibility validation, which is a separate feature.

## 15. Code interfaces and ownership changes

New modules:

```text
include/qw3/kvmem_storage.hpp
include/qw3/kvmem_io.hpp
include/qw3/kvmem_index_store.hpp
src/kvmem_storage.cpp
src/kvmem_io_sync.cpp
src/kvmem_io_threaded.cpp
src/kvmem_io_uring.cpp
src/kvmem_index_store.cpp
```

Primary interfaces:

```cpp
class KvStorageManager {
public:
    KvStorageSession create_session(const SessionCapacity &);
    StorageCapabilities capabilities() const;
    StorageTuning tuning() const;
};

class KvStorageSession {
public:
    BlockRecordRef allocate_block(BlockId, Generation);
    WriteTicket persist_completed_blocks(const PackedBlockBatch &);
    ReadTicket prefetch(const SelectionLoadPlan &);
    void release_suffix(BlockId first_removed);
};

class SelectionTransaction {
public:
    void submit_reads();
    void progress();
    void wait_layer(uint32_t layer);
    void commit();
    void rollback();
};
```

Changes:

- `QwenNativeBackend` owns one `KvStorageManager` and passes session handles to
  executors.
- `QwenExecutor` coordinates capture, selection, query replay, and GPU events,
  but no longer calls `fread/fwrite`.
- `KvMemBlock` replaces exclusive tier/slot fields with multi-residency and
  generation metadata.
- `KvMemStore::set_selection` becomes a non-mutating planner; a separate commit
  applies baked positions and working-set membership.
- `DeviceBackend` gains batched stripe pack, scatter+RoPE, and event APIs.
- `KvMemPrefetchState` holds tickets, slabs, and per-layer readiness rather than
  `std::future<std::vector<uint8_t>>`.
- permanent `kvmem_raw_k_chunks_` are removed after write-through is validated;
  bounded capture/staging buffers remain.
- `g_kbar_multi_` gains CPU-streamed placement.

## 16. Configuration

Existing user-facing capacity options remain:

```text
--kvmem-budget
--kvmem-gen-budget
--kvmem-block-tokens
--kvmem-cpu-gb
--kvmem-nvme-dir
--kvmem-nvme-gb
```

New stable options:

```text
--kvmem-storage-backend auto|sync|threaded|uring|gds
--kvmem-storage-direct-io auto|on|off
--kvmem-storage-calibrate auto|off|refresh
--kvmem-index-placement auto|gpu|cpu|ssd
```

Advanced overrides, primarily for experiments:

```text
--kvmem-storage-read-concurrency N
--kvmem-storage-layer-group N
--kvmem-storage-slab-mb N
--kvmem-storage-read-slabs N
--kvmem-storage-write-slabs N
--kvmem-storage-max-pending-write-mb N
```

`auto` is the production default. Every startup prints:

- detected device/topology;
- selected backend;
- direct-I/O status;
- stripe layout;
- application concurrency;
- slab and GPU staging bytes;
- GPU/CPU/SSD capacity split;
- estimated all-cold selection latency.

### 16.1 Recommended starting profiles

These commands describe the target interface after the rollout phases above;
the new storage flags are not all implemented yet.

Current-machine validation, 2M logical context:

```text
--ctx 2000000
--kvmem
--kvmem-budget 229376
--kvmem-gen-budget 32768
--kvmem-block-tokens 32
--kvmem-cpu-gb 72
--kvmem-nvme-dir /path/on/nvme/kvmem
--kvmem-nvme-gb 140
--kvmem-storage-backend auto
--kvmem-storage-direct-io auto
--kvmem-storage-calibrate auto
--kvmem-index-placement cpu
```

The measured manual overrides for reproducing the current Gen3 host profile
would be:

```text
--kvmem-storage-layer-group 4
--kvmem-storage-read-concurrency 16
--kvmem-storage-slab-mb 64
--kvmem-storage-read-slabs 3
--kvmem-storage-write-slabs 3
```

They should not be carried to another machine without recalibration.

Production 10M uses the same logical parameters but requires a dedicated SSD
with enough free space and an arena of approximately 700 GiB:

```text
--ctx 10000000
--kvmem-nvme-gb 700
```

The filesystem should retain at least 32 GiB outside that arena. The current
system SSD does not satisfy this production headroom requirement.

## 17. Performance model

For a fully cold 224 Ki-token window:

```text
7168 blocks * 2.125 MiB = 14.875 GiB
```

Lower bounds before overlap:

| Storage bandwidth | SSD read time |
|---:|---:|
| current measured 3.43 GiB/s | 4.34 s |
| representative 6.5 GiB/s | 2.29 s |
| representative 12 GiB/s | 1.24 s |

Pinned H2D at 20 GiB/s needs about 0.74 s for the same bytes and should be
mostly overlapped with SSD and scatter. On the current Gen3 SSD, the all-cold
floor is therefore around 4.3--5 seconds, not 10+ seconds.

Real repeated turns should be much better:

- retained GPU blocks transfer zero bytes;
- CPU cache hits avoid SSD;
- a 90% overlap between consecutive selected windows loads only about
  1.49 GiB;
- write-through removes normal stage-out writes;
- stripe readiness overlaps later reads with query replay.

Exact CPU-streamed softmax mean-K reads the 9.54 GiB index twice: once for
global max/denominator statistics and once for normalized score accumulation.
It therefore moves about 19.1 GiB. At 20 GiB/s H2D this is roughly 0.95 s
before dot-product and CPU-gather overhead, with transfer/scoring overlap.
When selection overlap is high, index scoring may become the dominant latency;
GPU-resident index placement is then valuable if VRAM permits.

## 18. Rollout plan

The executable A/B levels group related mechanisms by the latency source they
target:

| Runtime level | Current meaning |
|---|---|
| `kvmem_init` | frozen exclusive-tier compatibility path |
| `opt_1` | reduce SSD load volume with heat-aware CPU admission and eviction |
| `opt_2` | reduce stage-out latency with inclusive SSD backing, a bounded asynchronous positional-write queue, and recyclable pageable slabs |
| `opt_3` | reduce stage-in latency with coalesced positional reads and a two-slab preallocated/recycled SSD-read/H2D pipeline |

An unimplemented level must fail during configuration. It must never silently
fall back to an earlier path, because that would invalidate latency comparisons.
The current `opt_2` persists the existing immutable spill record (V plus any
configured tiered MTP payload) asynchronously when a block is staged out.
Persisting every completed prefill block before pressure selection, striped
layer readiness, and layer-by-layer replay overlap remain production follow-up
work rather than being implied by these three experimental levels.

The defaults are 64 MiB slabs and write queue depth 8. This bounds the
prewarmed pageable write pool at 512 MiB; `opt_3` adds two pinned read slabs
(128 MiB by default). They are portable
runtime settings, not device constants: `QW3_KVMEM_IO_SLAB_MIB` and
`QW3_KVMEM_WRITE_QUEUE_DEPTH` can be calibrated per host, are range-checked,
and preserve bounded memory/backpressure.

### Phase 0: preserve baseline

- freeze current sync behavior and sample-wise accuracy outputs;
- keep ongoing experiments on the legacy path;
- add feature-gated storage backend selection;
- retain `SyncKvIoBackend` as the correctness oracle.

### Phase 1: storage ownership and layout

- add global manager, unique sessions, record allocator, generations;
- implement striped fixed-offset files using positional synchronous I/O;
- remove `FILE *`, shared seek, `w+b` collision, and per-block `fflush`;
- add capacity preallocation and admission;
- keep answers byte-equivalent to the current immutable path.

### Phase 2: write-through raw K and V

- add GPU stripe pack kernel;
- persist completed prefill blocks asynchronously through write slabs;
- make SSD the raw-K authority;
- retain current raw-K CPU chunks behind a debug flag until equivalence passes;
- convert stage-out of clean blocks to page release.

### Phase 3: CPU cache and exact index

- implement pageable full-record cache and 2Q policy;
- implement CPU-chunked mean-K capture and tiled scorer;
- enforce the total 72 GiB host budget;
- validate selected block IDs exactly against the GPU-resident index.

### Phase 4: bulk GPU transfer

- add read slabs and device staging;
- add fused V scatter + raw-K RoPE scatter;
- eliminate per-block vectors and small CUDA copies;
- validate raw-K materialization byte-for-byte.

### Phase 5: asynchronous engine

- add worker-pool positional I/O;
- add optional `io_uring` fixed files/buffers;
- add priorities, cancellation, and per-session credits;
- overlap exact reads, clean release, H2D, scatter, and writeback.

### Phase 6: query-replay layer pipeline

- publish per-stripe readiness;
- add executor `wait_kv_ready(layer)`;
- overlap replay compute with later stripe loads;
- measure visible versus hidden I/O.

### Phase 7: production hardening

- fault injection and transaction rollback;
- multi-session isolation/admission;
- 2M then 10M soak;
- SSD thermal/endurance telemetry;
- optional GDS A/B only after host-staged path is stable.

Each phase is independently gated. The default switches from legacy to the new
path only after the corresponding correctness and performance gates pass.

## 19. Tests

Unit tests:

- stripe offsets, alignment, and record size for fp16/fp8/fp32;
- global slot allocation, session isolation, generation/COW reuse;
- state-machine ownership and last-copy guards;
- CPU cache admission/eviction;
- cancellation, reordered completions, short I/O, and error injection;
- transactional commit/rollback.

GPU tests:

- packed raw K/V equals direct source bytes;
- fused scatter produces the same V and one-shot RoPE K;
- partial last blocks;
- MTP local-position materialization;
- repeated remap threshold/raw refresh;
- no baked position reaches the model limit.

Integration tests:

- sync new backend versus current backend on deterministic selections;
- query replay and warm-prefix checkpoint cases;
- prefill-only multi-session continuation;
- LongMemEval-M ten-sample accuracy baseline;
- LongMemEval-S and AgentLongBench regression subsets;
- multiple simultaneous executors in one NVMe directory;
- truncate/replay while old writes are delayed;
- CPU and GPU hard-budget tests;
- 2M-token soak followed by 10M-token capacity test.

Performance tests:

- request size and concurrency sweep;
- direct versus buffered I/O;
- CPU gather, H2D/D2H, and scatter throughput;
- read/write mixed load at 200 MiB/s write-through;
- all-cold and high-overlap reselections;
- percentage of SSD, D2H, H2D, and scatter time hidden.

## 20. Acceptance criteria

Correctness:

- identical selected block IDs to the exact fp16 reference;
- raw-K scatter is byte-identical to one-shot materialization for the configured
  dtype;
- no missing/stale generation can be consumed;
- no silent retrieval or storage fallback;
- no accuracy regression on the fixed evaluation baselines.

Memory:

- one active GPU K and one V copy only;
- GPU staging included in the configured VRAM limit;
- total KVMem host allocation does not exceed `--kvmem-cpu-gb`;
- permanently pinned host memory below 1 GiB by default;
- SSD admission reserves required capacity before long prefill.

Performance:

- large-record demand reads reach at least 85% of the device's calibrated
  KVMem-sized read bandwidth;
- no per-block heap allocation, `fseek`, `fflush`, or CUDA memcpy in the steady
  path;
- steady 3K token/s prefill does not wait for SSD write-through;
- clean stage-out performs no SSD write;
- at least 70% of eligible SSD time is hidden by D2H/H2D/scatter/query replay;
- P95 and P99 reselection latency are reported, not only average throughput.

Observability per reselection:

- source/selected/retained/incoming/outgoing block counts;
- GPU, CPU, pending-slab, and SSD hit bytes;
- per-priority submitted/completed I/O;
- first/last completion latency and effective GiB/s;
- CPU gather and pinned H2D/D2H throughput;
- scatter/RoPE time;
- mean-K transfer/scoring time;
- overlap and visible wait;
- writeback lag and slab starvation;
- device temperature, throttling, AER/link downgrade where readable;
- chosen calibration key and tuning values.
