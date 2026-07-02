# KVMem Implementation Notes for Method Writing

This document summarizes the current KVMem implementation in the `qw3` codebase
as detailed source material for writing an academic method section. It is not
intended to be a concise paper draft. Instead, it records the mechanisms,
design invariants, equations, scheduling decisions, data structures, efficiency
arguments, limitations, and paper-relevant interpretation of the implementation.

The implementation is centered around three ideas:

1. Step-level memory scheduling: KVMem updates its memory state at structured
   execution points, especially after prefill and at configured reselection
   boundaries, rather than continuously recomputing memory decisions for every
   generated token.
2. KV-native memory retrieval: KVMem retrieves historical context blocks using
   model-internal query/key signals, with the default query-conditioned method
   using a mean-key index and a block-level softmax scorer.
3. Tiered KV memory management: KVMem stores the full recoverable KV state across
   a bounded GPU page pool, CPU pinned memory, and NVMe storage, while assembling
   only a selected working set for attention.

The most important implementation files are:

- `include/qw3/kvmem_store.hpp`: host-side block metadata, selection policy,
  remap plans, score fields, and tier tags.
- `src/kvmem_store.cpp`: host-side block registration, top-k/quota selection,
  and conversion from selected block IDs to stage-in/stage-out/remap plans.
- `src/qwen_executor.cpp`: executor integration, query capture, mean-key index
  construction, retrieval scoring, window assembly, re-RoPE, stage-in/out, and
  tiered I/O orchestration.
- `src/kernels_cuda.cu`: CUDA kernels for mean-key construction, softmax-over-page
  retrieval scoring, exact-mass scoring, de-RoPE, and batched re-RoPE.
- `include/qw3/pinned_kv_tier.hpp`: CPU pinned-memory tier metadata and LRU slot
  allocator.
- `include/qw3/nvme_kv_tier.hpp`: NVMe slot metadata and backing-file I/O.
- `include/qw3/device_backend.hpp`: backend abstraction for page-table operations,
  raw byte copies, KVMem kernels, and the reusable pinned host buffer pool.

## 1. Conceptual Model

KVMem treats a long agent/session context as a growing repository of KV blocks.
The full context may contain far more tokens than can remain active in GPU
memory or be attended to efficiently. Rather than summarizing or discarding old
tokens immediately, KVMem preserves their KV state and selects a bounded subset
of blocks to assemble into the active attention window.

The implementation is block-sparse at the level of the standard attention
layers. Recurrent/DeltaNet layers are not stored as KVMem blocks because they
maintain compact state rather than token-wise KV. The block selection is global:
all standard attention layers share the same selected block IDs. Per-layer
information is used for retrieval scoring, but the final working set is one
block list shared by all attention layers.

A KVMem block is a contiguous span of original token positions. The default
block size is controlled by `--kvmem-block-tokens`. The block size must be a
positive multiple of the physical KV page size. This invariant is enforced by
`QwenExecutor::configure_kvmem`. The reason is subtle but central: KVMem builds
the compressed attention window by reordering physical pages. If a block boundary
were not aligned to a KV page boundary, packing selected blocks contiguously in
the window could split one physical page across two logical window positions,
which would corrupt the addressing math.

KVMem distinguishes three positions for a block:

- The original position range: `[orig_pos_start, orig_pos_start + n_tokens)`.
- The current baked position: `baked_pos`, i.e. the position at which the block's
  K vectors are currently RoPE-encoded in the physical cache.
- The assigned window position: the compact position range used in the currently
  selected attention window.

After prefill, a newly registered block is canonical: `baked_pos == orig_pos_start`.
When a selected block is placed into a compact window, its keys may be re-RoPEd
in place from `baked_pos` to the new window position. This makes the selected
window stay within the model's normal positional range. Before a block is spilled
to CPU or NVMe, it is re-canonicalized back to its original position. Therefore
lower tiers store true-position KV bytes, not window-position bytes.

This gives the implementation an important invariant:

> GPU-resident blocks may be baked either at original positions or at current
> window positions, and `baked_pos` records which. CPU/NVMe-resident blocks are
> stored in canonical original-position form.

This invariant makes later stage-in and remap safe: a recalled block can always
be restored to GPU and then re-RoPEd from its recorded baked position to the next
selected window slot.

## 2. Core Data Structures

### 2.1 `KvMemBlock`

`KvMemBlock` is the metadata record for one historical block. The key fields are:

- `block_id`: dense ID in append order.
- `orig_pos_start`: the original first token position.
- `n_tokens`: number of tokens in the block, up to `block_tokens`.
- `tier`: current storage tier, one of `GPU`, `CPU`, or `SSD`.
- `gpu_slot`, `cpu_slot`, `nvme_slot`: physical handles, currently CPU/NVMe slots
  are actively used.
- `baked_pos`: the position at which the K vectors are currently RoPE-baked.
- `in_working_set`: whether the block is part of the current selected window.
- `profile_score`: cumulative window-local attention/profile signal.
- `retrieval_score`: global retrieval score from the KV-native retrieval path.
- `attn_score`: the score consumed by the selector. Depending on the method,
  this aliases either the profile score or the retrieval score.

The implementation keeps score fields separate so a quota selector can reserve
part of the budget for retrieval and part for profile/heavy-hitter retention.

### 2.2 `KvMemPlan`

`KvMemPlan` is produced by `KvMemStore::set_selection`. It contains:

- `stage_in`: selected blocks that are not currently resident in the GPU working
  set and may need to be brought back from CPU/NVMe.
- `stage_out`: GPU-resident blocks that are no longer selected and should be
  evicted from the bounded GPU pool if lower tiers are configured.
- `remaps`: a full ordered list of selected blocks, each with:
  - `block_id`
  - `n_tokens`
  - `from_base`: current baked position
  - `to_base`: assigned window position
  - `skip`: whether `from_base == to_base`
- `total_window_tokens`: sum of selected tokens, used as the query position at
  the end of the assembled window.

The plan is both a memory movement plan and a positional remap plan.

### 2.3 `KvMemStoreConfig`

The configuration includes:

- `block_tokens`: block granularity.
- `select_budget`: maximum selected window tokens.
- `sink_blocks`: prefix blocks always kept.
- `recent_blocks`: suffix blocks always kept; zero means derive a recent window
  from the budget.
- `select_method`: retrieval, H2O/profile, or recency.
- `select_policy`: top-k or quota.
- `retrieval_method`: `mean-k` or `per-token`.
- `update_mode`: interval or step.
- GPU memory ratio and watermarks.
- estimated GPU block capacity and block byte size.
- CPU and NVMe tier capacities.

The executor may adjust `select_budget` downward based on estimated GPU capacity,
so the selected working set fits the bounded page pool.

## 3. Step-Level Memory Scheduling

### 3.1 Problem

A naive memory manager could re-evaluate memory on every generated token. That
would be too expensive and unnecessary. In agent workloads, memory demand is
usually determined by the step prompt: the user question, recent tool outputs,
explicit file names, failed tests, or prior hypotheses. Once a step begins, the
memory target is relatively stable.

At the same time, memory cannot be managed only at process startup. The context
grows over time, and a bounded GPU KV pool may run out of pages during long
prefill. KVMem therefore needs a scheduling policy that is frequent enough to
bound memory and capture new blocks, but coarse enough to avoid per-token memory
planning overhead.

### 3.2 Current Scheduling Points

The implementation uses these scheduling/update points:

1. Before or during prefill:
   - Query-conditioned spans can be registered with `kvmem_set_query_span`.
   - The executor allocates query and mean-key buffers for the expected prompt
     length.
2. During prefill:
   - Newly written tokens are registered into `KvMemStore` through
     `kvmem_register_until` / `kvmem_register_append`.
   - Query rows inside the marked question span are captured and de-RoPEd.
   - Per-block content mean keys are captured from freshly produced K batches.
   - If a bounded GPU pool is close to exhaustion, KVMem triggers an in-prefill
     reselection/offload through `kvmem_maybe_prefill_offload`.
3. At the prefill-to-decode boundary:
   - KVMem performs the main selection and window assembly through
     `kvmem_reselect`, or the split `kvmem_prepare_reselect` /
     `kvmem_finish_reselect`.
4. During decode:
   - The assembled window page table is extended at the tail as new tokens are
     appended.
   - If interval update mode is active, KVMem can periodically reselect after
     a configured number of decode steps.
   - The profile/H2O scoring side channel can accumulate per-window-block
     relevance during decode.

This is why "step-level scheduling" is a good paper framing: the main expensive
memory decision happens at the boundary between a step's prompt prefill and its
decode, while lighter state maintenance happens during prefill/decode.

### 3.3 Block Registration

`KvMemStore::register_append` maintains a block table over the growing context.
It first extends the trailing partial block, then creates new blocks as needed.
Each new block receives:

- a dense block ID,
- its original starting position,
- the number of tokens captured,
- `baked_pos = orig_pos_start`.

This means the block table can grow monotonically with the session, while the
GPU residency of each block can change independently.

### 3.4 Reselect Prepare/Finish Split

The core reselection path is split:

- `kvmem_prepare_reselect`:
  - builds or uses the retrieval index,
  - scores blocks,
  - calls `pick_topk_blocks`,
  - produces a `KvMemPlan`,
  - stages out evicted blocks,
  - starts prefetch for selected non-resident blocks.
- `kvmem_finish_reselect`:
  - waits for prefetch,
  - assembles the window,
  - performs re-RoPE,
  - recomputes the window-local mean-key profile buffer.

The split is important for system performance. It allows the issue phase of
stage-in, especially NVMe reads and CPU-to-GPU copies, to overlap with other
work. The convenience function `kvmem_reselect` simply calls prepare and finish
back-to-back when no overlap is available.

### 3.5 Evict Before Stage-In

The implementation deliberately runs stage-out before stage-in. This is not
only a cleanliness choice; it is necessary for bounded GPU pools.

A query-conditioned reselect may resurrect many scattered historical blocks.
If the executor allocated GPU pages for those resurrected blocks before freeing
the pages of deselected blocks, the bounded pool could overflow. By evicting
first, the resident set after eviction is at most the selected window, and the
subsequent stage-in fits within the budget.

This ordering is especially relevant for paper discussion because it shows that
the memory scheduler is not merely choosing blocks; it enforces a resource
invariant required by the physical page allocator.

### 3.6 In-Prefill Offload

Long prefill can consume a large number of pages before reaching the step
boundary. `kvmem_maybe_prefill_offload(next_chunk_tokens)` checks whether the
bounded GPU page pool has enough free pages for the next prefill chunk plus a
small cushion of recent blocks. If not, it triggers a reselect during prefill.

This avoids a failure mode where the next chunk allocates pages in one burst and
exhausts the pool before the scheduler has a chance to evict. It also spreads
tiering cost over prefill rather than concentrating all eviction at the
prefill-to-decode boundary.

For method writing, this is the key argument:

> KVMem schedules memory at semantic step boundaries but also includes a
> resource-triggered in-prefill offload path to maintain the bounded-GPU
> invariant during very long prompt ingestion.

## 4. KV-Native Mean-K Retrieval

### 4.1 Problem

Text retrieval ranks stored text by lexical or embedding similarity. KVMem needs
a retrieval method over previously computed KV state. The direct approach would
score every query token against every historical key token across all standard
attention layers and heads. This is too expensive in both memory and compute.

The implementation therefore uses a mean-key block summary by default. The
summary should be:

- cheap enough to store for every historical block,
- computed in the same model-internal representation as attention,
- query-conditioned,
- able to rank blocks globally so dropped blocks can be resurrected.

### 4.2 Content-Frame Mean Key

For each block `b`, standard-attention layer `l`, and KV head `g`, KVMem stores
a mean key:

```text
kbar[l, b, g] = mean over tokens t in block b of deRoPE(K[l, t, g], position_t)
```

In formula form:

```text
\bar{k}_{l,b,g} = (1 / |b|) * sum_{t in b} deRoPE(k_{l,t,g}, p_t)
```

where:

- `p_t` is the position at which token `t` was actually RoPE-baked,
- `deRoPE` removes the positional phase,
- dimensions beyond `rope_dim` pass through unchanged.

The reason for de-RoPE is that selected blocks may later be re-RoPEd into
different compact window positions. Retrieval should compare content, not the
block's current window phase. The mean-key index is therefore stored in a
position-invariant content frame.

The CUDA kernels use the same sine/cosine convention as the forward RoPE path.
This matters because the code relies on cancellation of RoPE numerical error:
if a vector was originally baked with a certain trigonometric approximation,
de-RoPE with the same convention recovers a stable content representation.

### 4.3 Incremental Index Construction During Prefill

There are two ways to build content mean keys:

1. `block_kmean_content_paged_device`: read from the full paged KV repository,
   de-RoPE at original positions, and average.
2. `block_kmean_content_batch_device`: build the mean directly from the freshly
   produced contiguous `k_batch` during prefill.

The second path is the important current implementation for query-conditioned
retrieval. It exists because a post-prefill paged scan is only safe before any
block has been offloaded or re-RoPEd. In very long contexts, a bounded GPU pool
may force mid-prefill offload. If the index were built only from GPU-resident
pages after that point, it would miss offloaded blocks, making them impossible
to retrieve.

Therefore, when a query-conditioned span is active, KVMem pre-sizes the index
using the final prompt token count and captures each block as its K rows are
produced. Chunks are expected to be block-aligned. Each chunk-local block is
mapped to a global block ID, de-RoPEd at the actual bake position, averaged,
and written into the correct layer slice of `g_kbar_multi_`.

When all blocks have been captured, the executor publishes:

- `g_content_ready_ = true`,
- `g_kbar_multi_ready_ = true`,
- `g_kbar_multi_blocks_ = total_blocks`,
- `g_indexed_blocks_ = total_blocks`.

This means the retrieval scorer can rank all historical blocks even if their KV
bytes now live in CPU or NVMe.

### 4.4 Query Capture

The server marks a final user-message token span when query-conditioned KVMem is
enabled. The executor records this span through `kvmem_set_query_span(begin,
end, prompt_tokens)`.

During prefill, `kvmem_capture_query_multi` checks each chunk for overlap with
the query span. For the overlapping rows, it de-RoPEs the Q vectors into the
same content frame as the mean-key index. The captured buffer layout is:

```text
g_query_multi_[layer_slot, query_token, query_head, head_dim]
```

The query span is capped at 512 tokens in the current implementation. The cap is
a practical bound on retrieval overhead and memory. For most question-conditioned
agent steps, the final user question is short relative to the context.

### 4.5 Mean-K Softmax-Over-Pages Scoring

At the reselection boundary, the default query-conditioned retrieval method is
`kvmem_retrieval_score_mean_softmax`. It uses the captured query vectors and
the multi-layer mean-key index.

For each query token `m`, standard-attention layer `l`, query head `h`, and
candidate block `b`, it computes:

```text
logit[l, m, h, b] =
    (q[l, m, h] dot kbar[l, b, kv_head(h)]) / sqrt(head_dim)
```

Then it applies softmax over the block dimension:

```text
mass[l, m, h, b] = softmax_b(logit[l, m, h, b])
```

The final block score is accumulated as:

```text
score[b] = sum_m mean_l mean_h mass[l, m, h, b]
```

The current kernel sums over query tokens rather than averaging over them. Since
all blocks for a given request share the same query token count, this does not
change the ranking within that request.

The CUDA implementation is `block_attn_score_softmax_pages_kernel`:

- Grid: one CUDA block per `(layer, query token)`.
- For each query head, it computes logits for all memory blocks.
- The block logits live in dynamic shared memory.
- Thread 0 computes the max and sum for the softmax over blocks.
- Threads then atomic-add weighted softmax mass into `score[b]`.

The output `score[b]` is copied to host once and written into the block store as
`retrieval_score`. The selector then uses this score to choose blocks.

### 4.6 Why Softmax Over Blocks?

The scorer is not just a dot-product ranking. It forms a distribution over
blocks for each query token/head/layer. This is closer to the attention mechanism
that the model would use if it attended over the block summaries. It also makes
blocks compete globally: a block receives a high score only if it attracts mass
relative to the other blocks.

This is important for a paper explanation:

- Dot products alone give independent similarity scores.
- Softmax-over-pages turns those similarities into normalized attention mass.
- The method then ranks blocks by their expected share of query attention.

### 4.7 Relationship to Exact-Mass Retrieval

The code also supports `--kvmem-retrieval-method per-token`, implemented as
`kvmem_retrieval_score_exactmass`. This method stores raw de-RoPEd token keys:

```text
g_kraw_multi_[layer, token, kv_head, head_dim]
```

Then for each query token/head it computes a token-level softmax over all key
tokens and sums the resulting attention mass back into blocks:

```text
score[b] += sum_{t in b} softmax_t(q dot kraw[t])
```

This is more faithful because it avoids mean dilution. A single "needle" token
inside a block can still receive high softmax mass. However, it is much more
memory intensive. Comments in the implementation estimate the raw-key buffer
can be several GiB for long histories. The default mean-k path is therefore the
practical system choice.

### 4.8 Mean Dilution and Block Size Trade-Off

Mean-k retrieval is efficient, but it can dilute sparse evidence. If a block
contains many unrelated tokens and one crucial token, the block mean may not
represent the crucial token strongly enough. This is an algorithmic trade-off,
not a bug in the system mechanics.

The implementation exposes two main controls:

- smaller `block_tokens`, which reduces dilution but increases block count,
- larger `select_budget`, which keeps more candidate blocks.

The per-token exact-mass method is a stronger but heavier alternative.

For paper writing, it is worth distinguishing:

- KVMem's storage and window assembly machinery can be lossless under an
  all-block budget.
- Retrieval quality depends on the scoring approximation and budget.

This distinction is supported by the current evaluation notes in
`docs/kvmem_utility_eval_plan.md`: all-block KVMem matches full-context behavior
much more closely, while aggressive budget cuts expose retrieval recall limits.

### 4.9 Fallbacks

If query-conditioned mean-k is unavailable, KVMem falls back to other signals:

- A single last-token content scorer if the global content index and query are
  ready.
- Window-local H2O/profile scores if retrieval cannot run.
- Recency-only selection when no learned/model-internal signal is available.

Unsupported KV dtypes also matter. The mean-key/de-RoPE retrieval paths require
fp16 or fp32 KV. q8/fp8 KV cannot be meaningfully de-RoPEd/averaged in the
current implementation, so retrieval falls back to cheaper signals.

## 5. Selection Policy

### 5.1 Always-Kept Regions

The selector always preserves:

- sink blocks: a prefix region, usually the first block,
- recent blocks: a suffix region near the current tail.

The sink region preserves stable prefix information. The recent region prevents
the model from losing the immediate local continuation. If `recent_blocks` is
zero, the implementation derives a recent window from the budget, defaulting to
approximately one quarter of the budget, at least one block.

### 5.2 Top-K Policy

The default `TopK` policy fills the remaining budget with the highest
`attn_score` blocks among the middle blocks. Retrieval mode writes retrieval
scores into `attn_score`; H2O/profile mode writes cumulative profile heat into
`attn_score`; recency mode leaves no learned score, so the always-kept regions
dominate and ties resolve by recency.

The implementation uses `std::nth_element`, not a full sort, to select the top
blocks efficiently. The final selected list is emitted in ascending block ID
order so the window preserves chronological ordering.

### 5.3 Quota Policy

The `Quota` policy can allocate separate quotas for retrieval and profile
signals. For example, part of the budget can be filled by global mean-k
retrieval scores and part by window-local attention heat. This is useful because
retrieval and profile answer different questions:

- Retrieval can resurrect blocks that are currently off-window.
- Profile/H2O can retain blocks that the model has recently attended to.

If explicit quotas are not provided, the implementation derives a default split
from the remaining budget. Any leftover budget is filled using the combined
`attn_score` field.

## 6. Window Assembly and Re-RoPE

### 6.1 Problem

After block selection, the attention kernels need a compact sequence of selected
KV tokens. A naive implementation would copy selected KV tensors into a new
contiguous buffer. That would be expensive because every selected block contains
K and V for many layers and heads.

KVMem instead constructs a virtual compact window by reordering page indices.

### 6.2 Page-Table Assembly

For each selected block, `kvmem_assemble` maps the block's original logical pages
to their current physical GPU pages and appends those physical page IDs to
`window_pages_host_`. The result is a page table for the compact window:

```text
window logical page 0 -> physical page of selected block 0
window logical page 1 -> physical page of selected block 0
...
window logical page N -> physical page of selected block k
```

The selected KV bytes are not copied. The window page table aliases the same
physical GPU pages already used by the repository page table.

The page table is then uploaded to device memory. Attention uses this window
page table instead of the full repository page table.

### 6.3 Re-RoPE

The page table solves addressing, but selected keys may have been RoPE-baked at
their original positions or at a previous window position. To make the compact
window positionally consistent, KVMem re-RoPEs selected K tensors in place:

```text
K_current = RoPE(content, from_base + token_offset)
K_new     = RoPE(content, to_base + token_offset)
```

This can be implemented by:

```text
content = deRoPE(K_current, from_base + token_offset)
K_new   = RoPE(content, to_base + token_offset)
```

The implementation combines this into a remap kernel. It applies only to K, not
V, because values are not RoPE-positioned.

Each remap includes:

- `from_base`: current baked position,
- `to_base`: assigned compact window position,
- `n_tokens`: number of tokens in the block.

If `from_base == to_base`, the remap is skipped.

### 6.4 Batched Re-RoPE

The efficient path batches all moved blocks for a layer into one kernel launch.
The executor collects arrays:

- `to_base[]`
- `from_base[]`
- `n_tokens[]`

Then it launches `rope_block_remap_paged_batched_device` once per standard
attention layer. The CUDA grid covers:

```text
(max_n_tokens, n_kv_heads, n_moved_blocks)
```

This avoids a launch storm of one kernel per `(layer, block)` pair.

### 6.5 MTP Window Mirror

When MTP speculative decoding is active and KVMem tiering is enabled for the MTP
KV cache, the executor builds a separate MTP window page table over the MTP
cache. MTP tracks its own baked positions because during prefill the main cache
and MTP cache can diverge: main K may be compressed into a KVMem window while
MTP remains canonical or is primed lazily.

The implementation therefore maintains `mtp_baked_pos_` separately and performs
MTP re-RoPE using this MTP-specific baked position state. This avoids incorrectly
rotating MTP keys based on the main cache's baked position.

## 7. Tiered KV Memory Management

### 7.1 Problem

A long session can contain far more KV than fits on GPU. Even if full KV could
fit, attending over all of it would make decode cost grow with total history.
KVMem needs to keep only a bounded working set GPU-resident while preserving
the ability to recall older KV blocks.

The current implementation uses three tiers:

1. GPU bounded page pool.
2. CPU pinned memory.
3. NVMe backing file.

### 7.2 Block Byte Layout

One tier slot stores one block's KV bytes across all standard-attention layers.
The layout written by `kvmem_copy_block_to_host_ptr` is:

```text
for each standard attention layer:
  for each page in the block:
    K page bytes
    V page bytes
optional MTP segment:
  for each page:
    MTP K page bytes
    MTP V page bytes
```

The estimated block byte size is computed from:

```text
block_tokens * n_kv_heads * head_dim * num_layers * 2(K,V) * elem_bytes
```

For q8 KV, row scale overhead is included in the estimate. However, CPU/NVMe
offload currently rejects 1-byte KV dtypes in `kvmem_kv_page_bytes`, so practical
tiered offload currently targets fp16/fp32 KV.

### 7.3 GPU Bounded Page Pool

When KVMem is configured with spill capacity and the estimated full context
would exceed the GPU budget, the executor creates a local page pool:

- The page pool has a fixed number of physical KV pages.
- `kv_pages_` maps logical token pages to physical pages allocated from this pool.
- The K/V cache tensors are allocated only for the bounded number of physical
  slots.

The number of GPU-resident blocks is computed from:

- total GPU memory,
- requested `gpu_memory_ratio`,
- current used GPU memory,
- scratch reserve,
- estimated bytes per block.

The implementation reserves scratch memory because prefill attention workspaces
and matmul staging buffers also consume GPU memory. This prevents the KV pool
from being sized too aggressively.

### 7.4 CPU Pinned Tier

The CPU tier is a fixed number of slots in page-locked host memory. The metadata
class `PinnedKvTier` tracks:

- `slot_count`,
- free slot list,
- `block_id -> slot`,
- LRU order.

The class does not own the backing memory and does not issue copies. The executor
owns or borrows the actual `HostBuffer`. This separation keeps tier placement
logic testable without CUDA.

Pinned memory is important because D2H/H2D raw byte copies can be asynchronous
on a copy stream. A pageable buffer would force the driver to serialize through
an internal bounce buffer, increasing stage-out and stage-in cost.

For continuous batching, a `HostTierBufferPool` can recycle pinned buffers keyed
by exact byte size. This avoids paying large `cudaHostAlloc` costs per request.
The pool is safe because each executor starts with empty tier metadata and only
reads slots it wrote itself.

### 7.5 NVMe Tier

The NVMe tier stores fixed-size block slots in a backing file:

```text
<nvme_dir>/qw3_kvmem_nvme.bin
```

`NvmeKvTier` tracks:

- `block_id -> slot`,
- free slot list,
- LRU order,
- synchronous `write_block` and `read_block`.

Although the tier's file I/O functions are synchronous, the executor performs
NVMe reads in a background `std::async` during prefetch. This gives overlap
between disk read and other work.

### 7.6 Stage-Out: GPU to CPU/NVMe

Stage-out evicts blocks that are no longer selected.

The steps are:

1. Check that the block's pages are GPU-resident.
2. Compute the expected spill byte size.
3. Ensure a pinned single-block staging buffer.
4. Canonicalize the block:
   - if the main K cache is baked at a window position, re-RoPE it back to the
     original position;
   - if MTP tiering is active, canonicalize MTP K using its own baked-position
     tracking.
5. Begin a device-to-host transfer on the KV copy stream.
6. Copy each K and V page into the pinned staging buffer.
7. Wait for the copy stream.
8. Place the block into the CPU tier if available.
9. If the CPU tier is full and NVMe is available, evict the CPU LRU victim to
   NVMe, then reuse that CPU slot.
10. If no CPU tier exists, write directly to NVMe.
11. Release the block's GPU logical pages back to the bounded pool.
12. Update the block store tier metadata.

The canonicalization step is critical for correctness. Without it, lower tiers
would contain window-relative K vectors. If the same block were later recalled
into a different window slot, the executor would not know how to reconstruct
the correct RoPE frame. Storing canonical bytes makes lower tiers independent
of previous window layouts.

### 7.7 Stage-In: CPU/NVMe to GPU

Stage-in is implemented as a prefetch protocol.

`kvmem_start_prefetch`:

- opens a transfer-to-device phase,
- skips blocks whose pages are already resident,
- for CPU blocks:
  - finds the CPU slot,
  - ensures target logical GPU pages are resident,
  - queues asynchronous H2D copies from pinned CPU memory;
- for NVMe blocks:
  - records a read request,
  - allocates a host buffer,
  - launches a background thread to read the bytes.

`kvmem_finish_prefetch`:

- waits for the NVMe background read if present,
- ensures GPU pages for NVMe blocks,
- queues H2D copies for those buffers,
- waits for the KV copy stream,
- releases CPU/NVMe slots,
- marks blocks as GPU-resident.

This split allows the scheduler to issue prefetch early and wait only when the
assembled window is required.

### 7.8 Copy Stream Ordering

The backend provides raw byte copy operations and transfer synchronization. The
important system design is that KV transfers use a dedicated copy stream rather
than the main execution stream. Transfer setup records appropriate ordering so
device-to-host reads do not start before the producing kernels complete. The
executor then waits on the copy stream only when it needs the bytes.

For paper writing, the concise statement is:

> KVMem separates compute and KV transfer streams, allowing pinned-memory copies
> and NVMe reads to overlap with independent GPU work whenever the caller uses
> the prepare/finish reselection split.

## 8. Decode-Time Window Growth

After a window is assembled, new decode tokens are appended at the window tail.
The true KV cache also appends these tokens at the current true sequence
position. KVMem extends `window_pages_host_` by aliasing the physical pages of
the true tail. No copy is needed because the same physical page can be addressed
through both:

- the repository page table, and
- the compact window page table.

For batched verify/MTP paths, `kvmem_extend_window_for_decode_n` grows the
window for multiple tokens at once and syncs the page table once. The method
uses the fixed offset between true cache position and compact window position
to map window pages to true pages.

This is part of why the decode path stays efficient: after assembly, adding new
tokens requires only occasional page-table extension, not rebuilding the full
window every token.

## 9. H2O/Profile Scoring Side Channel

Although the paper focus may be mean-k retrieval, the implementation also
contains a lightweight profile signal similar in spirit to heavy-hitter
retention.

After a window is assembled, `kvmem_recompute_kbar` computes a mean baked key
for each selected window block at a representative standard-attention layer.
This differs from content mean-k retrieval:

- It is computed only for the current window.
- It uses baked window-frame K, not content-frame de-RoPEd K.
- It is used to estimate which current window blocks the model attends to during
  decode.

At each decode step and representative layer, `kvmem_score_current_step` dots
the current RoPE-baked Q against each window block's mean baked K and accumulates
a nonnegative score. At the next reselect boundary, `kvmem_drain_scores` copies
the accumulator to host and folds it into `profile_score`.

This signal cannot resurrect blocks that are no longer in the window, because it
only scores the current window. Retrieval can resurrect old blocks because it
scores all indexed historical blocks. The quota policy can combine both.

## 10. Efficiency Arguments for a Method Section

The implementation contains several system-level efficiency mechanisms that are
worth emphasizing in a paper.

### 10.1 Retrieval Efficiency

Mean-k retrieval stores block summaries instead of raw token keys:

```text
O(num_layers * num_blocks * n_kv_heads * head_dim)
```

instead of:

```text
O(num_layers * num_tokens * n_kv_heads * head_dim)
```

This makes query-conditioned retrieval feasible for long histories. The scorer
uses one fused CUDA launch for the softmax-over-pages path and one host copy of
the final block scores.

### 10.2 Indexing During Prefill

Building the mean-key index during prefill avoids a second full pass over the
KV cache. It also covers blocks that are later offloaded. This is both more
efficient and more correct for a bounded GPU pool.

### 10.3 No-Copy Window Assembly

The selected working set is assembled through page-table aliasing. KVMem does
not copy selected K/V tensors into a new dense buffer. It uploads a compact page
index list and reuses existing physical pages.

### 10.4 Batched Re-RoPE

Only moved blocks are re-RoPEd. Blocks already baked at their assigned window
position are skipped. All moved blocks are processed by one batched kernel per
standard-attention layer.

### 10.5 Canonical Lower-Tier Storage

CPU/NVMe tiers store canonical true-position KV bytes. This avoids accumulating
unknown positional transformations across multiple evictions and recalls.

### 10.6 Pinned CPU Memory

Pinned CPU memory enables asynchronous raw byte copies over a dedicated copy
stream. This is a major practical difference from pageable host staging.

### 10.7 Split Prefetch

Stage-in is split into issue and finish phases. NVMe reads run in a background
thread; CPU H2D copies are queued asynchronously. The caller can overlap these
operations with independent compute.

### 10.8 Proactive In-Prefill Offload

KVMem does not wait until prefill completes to evict cold blocks. It monitors
free pages and triggers reselect/offload before the next prefill chunk can
exhaust the bounded pool.

## 11. Correctness and Invariants

Important invariants:

1. `block_tokens` must be a multiple of KV page size.
2. Block IDs are chronological and dense.
3. Selected windows are assembled in ascending block ID order.
4. The final window is compact: selected blocks are packed from position 0.
5. `baked_pos` records the current RoPE frame of GPU-resident main K.
6. MTP has separate baked-position tracking.
7. CPU/NVMe tiers store canonical original-position K/V bytes.
8. V does not require re-RoPE; only K is position-rotated.
9. Stage-out must run before stage-in for bounded pools.
10. Retrieval index readiness gates query-conditioned scoring; unavailable
    retrieval falls back rather than corrupting selection.

These invariants are useful for a paper because they show that the design is
not simply approximate retrieval over text, but a careful memory system that
preserves positional semantics while moving KV across tiers.

## 12. Limitations and Open Design Trade-Offs

### 12.1 Mean-Key Dilution

Mean-k can under-rank blocks where the relevant evidence is sparse. This is the
primary algorithmic limitation of the current default retrieval method. Smaller
blocks or larger budgets can reduce the problem; exact-mass retrieval addresses
it more directly but uses much more memory.

### 12.2 KV Dtype Restrictions

The tiered CPU/NVMe offload path currently rejects 1-byte KV dtypes. The
retrieval mean-key/de-RoPE path also expects fp16/fp32 K. q8/fp8 KV can still
run other paths, but KV-native retrieval may fall back.

### 12.3 Synchronous NVMe Writes

NVMe reads are overlapped during stage-in through background threads, but writes
in `NvmeKvTier::write_block` are synchronous. In current usage, stage-out is
still bounded by this design. More advanced asynchronous I/O could improve
stage-out behavior.

### 12.4 Host-Side Selector

Block scores are copied back to host and selection is performed on the CPU.
This is simple and flexible, but it means each reselect has at least one D2H
score transfer. For hundreds or thousands of blocks this is small, but a fully
device-side selector could reduce overhead.

### 12.5 Global Block Selection

The selected block set is global across standard-attention layers. This reduces
complexity and makes window assembly straightforward, but it cannot exploit
layer-specific memory needs. The retrieval score can use multiple layers, but
the final selection remains one shared set.

### 12.6 Full-Fidelity KV vs Text Fallback

The implementation focuses on preserving KV bytes across tiers. A broader
workspace memory system may also store text and metadata as the source of truth
for audit, deletion, and compatibility. The current KVMem implementation can be
described as the KV-native substrate rather than the complete text+KV memory
product.

## 13. Suggested Paper Framing

The method section can be organized around the following problem-solution flow.

### 13.1 Step-Level Memory Scheduling

Problem:

Long-running agents generate memory pressure at step boundaries, but token-level
memory decisions would be too expensive. A bounded GPU KV pool also needs
resource-triggered protection during long prefill.

Solution:

Schedule major memory decisions at step/reselect boundaries. Register new
blocks as context grows, capture retrieval features during prefill, trigger
in-prefill offload when page headroom is low, and assemble a selected window at
the prefill-to-decode boundary.

Key implementation points:

- `register_append`
- `kvmem_set_query_span`
- `kvmem_capture_query_multi`
- `kvmem_capture_kbar_multi`
- `kvmem_maybe_prefill_offload`
- `kvmem_prepare_reselect`
- `kvmem_finish_reselect`

### 13.2 KV-Native Memory Retrieval

Problem:

Text similarity does not directly measure whether the model's current query
would attend to a historical KV block. Exact token-level KV scoring is expensive.

Solution:

Construct a content-frame per-block mean-key index and score all blocks by
query-conditioned softmax mass over mean keys. Use all standard-attention layers
and all captured query tokens by default. Optionally support raw-key exact-mass
retrieval for higher fidelity.

Key equations:

```text
\bar{k}_{l,b,g} =
    (1 / |b|) sum_{t in b} deRoPE(k_{l,t,g}, p_t)
```

```text
r_b =
    sum_m mean_l mean_h softmax_{b'}
    (q_{l,m,h}^T \bar{k}_{l,b',g(h)} / sqrt(d))_b
```

Key implementation points:

- `kvmem_capture_kbar_multi`
- `kvmem_capture_query_multi`
- `block_kmean_content_batch_kernel`
- `block_attn_score_softmax_pages_kernel`
- `kvmem_retrieval_score_mean_softmax`

### 13.3 Tiered KV Memory Management

Problem:

The full KV history can exceed GPU memory, but discarding it loses the ability
to reuse computation. Moving selected KV blocks naively would also be too slow.

Solution:

Use a bounded GPU page pool for the active working set, CPU pinned slots for
warm blocks, and NVMe slots for cold blocks. Store lower-tier blocks in
canonical true-position form. Assemble selected windows by page-table aliasing
and in-place batched re-RoPE rather than by copying selected KV into a dense
buffer.

Key implementation points:

- `configure_kvmem`
- `kvmem_stage_out`
- `kvmem_start_prefetch`
- `kvmem_finish_prefetch`
- `kvmem_assemble`
- `rope_block_remap_paged_batched_device`
- `PinnedKvTier`
- `NvmeKvTier`

## 14. One-Sentence Summary

KVMem implements a block-sparse, KV-native memory system in which each agent step
selects a bounded set of historical KV blocks using query-conditioned mean-key
retrieval, assembles them as a compact attention window through page-table
aliasing and in-place re-RoPE, and preserves the remaining KV state across a
GPU/CPU/NVMe hierarchy with canonical lower-tier storage.
