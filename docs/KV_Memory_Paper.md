# KV Memory Paper

Block-sparse KV attention for a single long context in qw3: paged-KV block
selection, essential-KV assessment via cumulative attention, and the runtime KV
state machine. This document describes the implementation as it currently stands
(tasks #37–#40 and #43 complete; #41 tiering and #42 CLI/regression-guard
remaining).

## Problem

An agent session accumulates a single long context (cumulatively up to ~1M
tokens) that exceeds the model's attention context budget (e.g. 128k). Each
selection round, a subset of context blocks (within the budget) is chosen and
remapped into a contiguous attention window. This is **KV reuse, not length
extrapolation**: the active window length stays within the trained regime, and
RoPE is re-baked so relative positions remain in-distribution.

Two model facts make the problem tractable in qw3 (Qwen3.6-27B):

1. Only ~1/4 of layers are standard RoPE attention layers
   (`is_standard_attention_layer`, `full_attention_interval=4`). Only these
   layers have KV that grows with context. The other 3/4 are DeltaNet recurrent
   layers with O(1) state and no RoPE — they are **untouched** by block-sparse
   and act as a global summary while sparse attention provides precise recall.
2. The decode attention kernel already consumes an arbitrary page-index list, so
   block-sparse attention is "feed the kernel only the selected blocks' physical
   pages" — no attention-kernel change required.

A master switch (`--kvmem`, default OFF) isolates the entire feature;
when off the forward path is byte-identical to the pre-feature code. All other
parameters are CLI flags (no environment variables for runtime config).

---

## 1. KV selector over paged KV

The selector is pure host logic in `include/qw3/kvmem_store.hpp` and
`src/kvmem_store.cpp`, layered on top of the existing paged KV cache. It owns
no GPU memory and issues no copies.

### Block model

The growing context is sliced into fixed-size blocks of `block_tokens` tokens
(default 128), required to be a multiple of the 16-token physical KV page so a
block boundary never splits a physical page across two window slots.

```
struct KvMemBlock {
    uint32_t block_id;        // dense append-order index 0,1,2,...
    uint32_t orig_pos_start;  // first original token position in block
    uint32_t n_tokens;        // <= block_tokens
    KvTier   tier;            // GPU | CPU | SSD (logical; tiering is #41)
    int64_t  baked_pos;       // window position K is CURRENTLY baked to (in place)
    bool     in_working_set;
    double   attn_score;      // cumulative attention heat (selection signal, #40)
};
```

`register_append(n)` extends the trailing partial block first, then spills into
new blocks. After prefill a block is baked at its true sequential position
(`baked_pos == orig_pos_start`).

### Selection → plan

`set_selection(ids)` sorts/dedupes the requested IDs, drops out-of-range entries,
diffs against the current working set, and produces a `KvMemPlan`:

```
struct KvMemRemap {
    uint32_t block_id;
    uint32_t n_tokens;
    int32_t  from_base;   // current bake position (de-rotate source)
    int32_t  to_base;     // assigned in-window first-token position
    bool     skip;        // from_base == to_base: no kernel needed
};
struct KvMemPlan {
    std::vector<uint32_t>     stage_in;   // selected, not currently resident
    std::vector<uint32_t>     stage_out;  // resident, no longer selected
    std::vector<KvMemRemap> remaps;     // window order
    uint32_t total_window_tokens;
};
```

Selected blocks pack contiguously from window position 0 in ascending block-id
order (sink first ... recent last), so window order is chronological and
deterministic. Each block's `baked_pos` is updated to its new window slot; blocks
already sitting in the right slot get `skip=true` (the dominant case across
rounds with stable selection).

### No-copy window assembly

`QwenExecutor::kvmem_assemble` (`src/qwen_executor.cpp`) builds the window
without copying KV:

- **Window page table** = each selected block's *original physical pages*, in
  window order, packed contiguously. These alias the same `kv_pages_` slots the
  block already occupies — the window is a reordering of pointers, uploaded to
  `window_pages_device_`.
- **In-place re-RoPE**: for every standard-attention layer, each non-skip block
  is re-baked from `from_base` to `to_base` via `rope_block_remap_paged_device`.
  This **de-rotates** by the original position and **re-rotates** to the new
  position using the *same* `__sincosf` as the original bake, so the
  range-reduction error cancels and the result is lossless to ~1 fp16 ULP.
  (Single-delta rotation is intentionally avoided — see the re-RoPE note below.)
- `window_query_pos_ = total_window_tokens` is the query position for decode.

### Decode substitution

In `forward_one_token`, when `kvmem_active_`, the standard-attention block
substitutes the window for the live cache:

```
const bool bs = kvmem_active_;
if (bs) kvmem_extend_window_for_decode();
const uint32_t attn_pos      = bs ? window_query_pos_ : position_;
const DeviceTensor &pages_dev = bs ? *window_pages_device_ : kv_page_indices_device();
const uint32_t pages_count    = bs ? window_page_count_    : kv_page_count();
```

These feed the Q/K RoPE bake position, the paged KV append, and the paged decode
attention. Under an identity (all-fit) selection, every window slot equals the
block's true baked position, so every remap skips, the window page list equals
`kv_pages_.pages` verbatim, and `window_query_pos_ == position_` — decode is
byte-identical to the plain path.

### Re-RoPE correctness (carried from #38)

`rope_block_remap_paged_kernel` de-rotates by `R~(p_orig)^T` then re-rotates by
`R~(p_new)`, both with the same `__sincosf` as production's `rope_partial_kernel`.
The naive single-delta rotation `R((p_new−p_orig)·f)` on the stored key fails at
large `p_orig` (byte-error 0.05–0.15 at `p_orig`~1M) because the stored key was
baked with `__sincosf` at a huge angle whose float range reduction loses
precision; `__sincosf` is not a homomorphism at large angles, so the stale error
stays baked in. De-rotating with the identical `__sincosf(ang_o)` values cancels
that error exactly. Gold standard: `tests/rope_remap_parity.cu` (device-vs-device)
and `tests/kvmem_enabled_assembly.cu` (paged, scrambled page table), both green.

---

## 2. Essential-KV assessment (cumulative attention, #40)

This decides which blocks are "essential" and should survive reselection. The
design is **low-intrusion**: it does not touch the FlashInfer attention kernel or
alter its numerical output. Flash attention is online-softmax and never
materializes the score matrix, so a lightweight side channel estimates per-block
attention heat instead.

### Representative K vector

`block_kmean_paged_kernel` (`src/kernels_cuda.cu`) computes, per window block and
per KV head, the **mean baked K** over the block's tokens — a single
representative vector `k̄_block[kv_head]` of length `head_dim`. It runs at one
representative standard-attention layer (`bs_score_layer_` = the first standard
layer). Because selection is global (all attention layers share the same selected
block IDs), a single cross-layer representative is consistent and avoids the
~256MB cost of keeping per-layer means (16 layers × blocks × heads × head_dim).

It reads the **just-re-RoPE'd** window K, so `k̄` lives in the same window frame as
the Q it will be scored against. It is rebuilt once per reselect interval (cost
amortized over the interval), not per step. Output layout:
`kbar[n_blocks, n_kv_heads, head_dim]` (fp32).

### Per-step score

`block_attn_score_step_kernel` runs once per decode step at the representative
layer and adds to a GPU-resident accumulator:

```
accum[w] += Σ_qhead  max(0, scale · (q[qhead] · k̄_w[kv_head]))
```

where `kv_head = qhead / (n_heads / n_kv_heads)`. This is an unnormalized,
monotone proxy for "how much this block is attended to". Implementation: one CUDA
block per window block, `blockDim = head_dim` threads doing a shared-memory dot
reduction per query head; no cross-block atomics (each block writes its own
`accum[w]`), no per-step device→host copy. Cost is one extra global launch per
step (<0.1% of step time; decode throughput unchanged at ~45 tok/s).

### Drain and rank

`kvmem_drain_scores` runs **only at the reselect boundary** (every
`kvmem_interval` steps): it copies the accumulator to host (~4KB), folds it
into `KvMemStore::attn_score` indexed by block-id, then resets for the next
interval. `pick_topk_blocks` then ranks by `attn_score` descending (tie-break by
recency for determinism), always keeping the first `sink_blocks` and last
`recent_blocks` blocks.

```
kvmem_reselect():
    kvmem_drain_scores();                          // fold heat into attn_score
    plan = set_selection(pick_topk_blocks());             // rank by heat
    kvmem_assemble(plan);                          // window + re-RoPE
    kvmem_recompute_kbar();                        // rebuild k̄, reset accum
```

### Backend interface

Two virtuals on `DeviceBackend` (`include/qw3/device_backend.hpp`):
`block_kmean_paged_device` and `block_attn_score_step_device`. The CUDA backend
overrides both; the default base implementations are no-ops returning an error
that the executor tolerates — so a backend without overrides (or a q8/fp8 KV
cache, where averaging K is meaningless) silently falls back to recency-only
selection.

### Semantic property

A block must be **in the window** to accumulate score, so this is a heavy-hitter
**retention** signal: it keeps heavily-attended blocks resident across
reselections, but it cannot resurrect a block that has already been dropped from
the window. This is the standard heavy-hitter (H2O-style) eviction behavior. The
sink and recent windows in `pick_topk_blocks` guarantee a newly-entered block
survives its first interval, giving it a chance to accumulate score before it can
be evicted.

---

## 2b. Global content-frame KV retrieval (#48/#49)

The cumulative-attention signal above is a **retention** policy: a block must be
inside the active window to accrue heat, so a block already dropped from the
window can never come back, no matter how relevant the current query is. Global
retrieval lifts that limit — it scores **every historical block** each interval
by content similarity to the current query, so a dropped middle block is
**resurrected** when the query matches it (Quest / InfLLM-style retrieval rather
than H2O-style retention).

### Content-frame index

The naive approach — dot the current RoPE-baked query against each block's
RoPE-baked mean key — fails: the two live in different RoPE frames (the block at
its true position, the query at the window tail), so the raw dot is dominated by
positional phase and far blocks always lose. Instead both operands are moved to
the **content frame** (de-RoPE'd to position 0):

- `block_kmean_content_paged_kernel` reads each block's K through the **full
  repository page table** at its **true** positions and **de-RoPEs** every token
  by `(orig_pos_start + tok)` before averaging, recovering the position-invariant
  raw mean key `k̄_raw[block, kv_head]`. De-rotation uses the **same `__sincosf`**
  as the original bake, so the huge-position range-reduction error cancels (same
  reason the assembly re-RoPE de-rotates rather than single-delta rotates).
- `derope_query_kernel` de-RoPEs the current decode query by `window_query_pos_`
  into a content-frame query `q_raw[head, dim]`.

`q_raw · k̄_raw` is then a position-invariant content similarity.

### Build once, immutable

The content mean is position-invariant, and the only moment every block's K sits
at its true baked position is **right after prefill, before the first assembly
re-RoPE** (assembly moves selected blocks into window slots in place). So the
index is built **once** at the first reselect (`kvmem_build_content_index`,
gated on `!kvmem_active_`) and kept for the whole session — it covers all
prefill blocks. Decode-generated blocks are not indexed; they are recency-kept
(the sink/recent windows in `pick_topk_blocks` always keep the tail).

### Scoring and selection

- **Per step** at the representative layer: `kvmem_snapshot_content_query`
  de-RoPEs the live query into `g_query_content_`.
- **Per interval** (`kvmem_retrieval_score`): zero the score buffer (so the
  reused `block_attn_score_step` kernel **overwrites** rather than accumulates —
  retrieval re-ranks fresh each interval), score all indexed blocks against the
  content query, D2H, and `set_attn_scores` (overwrite, not `accumulate_attn`).
  `pick_topk_blocks` already ranks over **all** blocks, so this is the only host
  change — and it can now select a block that is not currently resident.

`kvmem_reselect` prefers retrieval and falls back to the window-local heat
fold (`kvmem_drain_scores`) only when the index/query is not live — e.g. a
q8/fp8 cache that can't be de-RoPE'd, or the first post-prefill selection before
any query exists. An internal diagnostic `QW3_KVMEM_RETRIEVAL_DISABLE=1` (default
off) forces the recency-only path for A/B validation at identical cache quality.

### Validation

1. **Zero-impact under identity selection** preserved: with a large
   (all-fit) budget, decode is byte-identical to plain despite the content index
   build, per-step query de-RoPE, and per-interval retrieval scoring all running
   (Count-to-20 verbatim).
2. **Resurrection of a dropped block.** A passkey (`ZEPHYR-7`) buried in the
   middle of a ~18k-token context, with a tight 64-block window over ~1016 blocks
   (`--kvmem-block-tokens 16 --kvmem-budget 1024 --kvmem-interval 1`,
   sink=1/recent=4): recency-only **fails** the recall (`Z-56` — the middle block
   is far outside the recency window) while global retrieval **recalls**
   (`Z-EPHYR-7`). Only the selection policy differs (both fp16). The trace
   (`QW3_KVMEM_TRACE=1`) shows the codeword-region block scoring into
   the top-k and being selected even though it sits deep in the middle.

---

## 3. Runtime KV state

### Live cache is the repository

There is no separate "live" page pool. The stored `kv_pages_` cache *is* the
block repository (no-copy design). Per decode token:

- `ensure_kv_pages(position_, 1)` grows the live cache.
- `kvmem_extend_window_for_decode` appends the trailing physical page to
  the window page table so the new token lands at window slot `window_query_pos_`.
  In the no-copy design this trailing window slot aliases the same physical page
  `kv_pages_` just grew.
- `position_` (true sequential position) and `window_query_pos_` (window position)
  advance in lockstep; they are equal under identity selection.

### Reselect cadence

In the backend `generate_plain` path (`src/qwen_native_backend.cpp`):

- Executor construction reads the CLI options into a `KvMemStoreConfig`, calls
  `set_kvmem_enabled(true)` and `configure_kvmem(cfg)`
  (which validates `block_tokens % page_size == 0`).
- After prefill: `kvmem_register_append(prompt_len)` then
  `kvmem_reselect()` builds the first working set (under the default
  all-fit budget this selects every block → identity → byte-identical decode).
- Per committed decode token: `kvmem_register_append(1)`; on the
  `kvmem_interval` boundary, `kvmem_reselect()`.

### Session reset

`reset_state` and `configure_kvmem` drop the per-session working set,
window page table, and the live score interval. Device buffers (`bs_kbar_`,
`bs_score_accum_`, `bs_win_base_dev_`, `bs_blk_tokens_dev_`) stay allocated and
grow lazily by block capacity. `bs_score_layer_` is model-fixed and kept.

### Tiering

`PinnedKvTier` (`include/qw3/pinned_kv_tier.hpp`) is a host slot allocator: a
fixed-count slab, a `block_id ↔ slot` residency map, LRU victim ordering, and
explicit eviction/release hooks. `NvmeKvTier`
(`include/qw3/nvme_kv_tier.hpp`) owns a raw backing file with the same fixed-slot
residency model.

When `--kvmem-cpu-bytes` is positive, `QwenExecutor` stages unselected
GPU-resident blocks out of the KV cache into the CPU tier and marks their
logical pages non-resident. If CPU slots are full and `--kvmem-nvme-dir` /
`--kvmem-nvme-bytes` are configured, the CPU LRU block is written to NVMe before
the slot is reused. When a selected block is not GPU-resident, it is read back
from CPU or NVMe, assigned fresh GPU physical pages for its original logical
range, copied into the KV cache, then assembled into the active window.

Tier backing is canonicalized for accuracy: before a GPU-resident block is
copied to CPU/NVMe, K is re-RoPE'd back from its current `baked_pos` to
`orig_pos_start`. V is position-invariant and is copied as-is. This keeps the
CPU/NVMe copy in the true-position frame, avoiding long-run accumulation from
repeated window-frame spills. Stage-in then starts from this canonical frame and
normal window assembly re-RoPEs to the selected window slot.

`QW3_KVMEM_TIER_TRACE=1` emits diagnostic `stage_out`, `cpu_evict`,
`canonicalize`, and `stage_in` lines. The model-gated E2E runner uses this to
assert that tiered sparse runs actually exercise GPU→CPU/NVMe stage-out and
CPU/NVMe→GPU stage-in.

The copy path is still synchronous. Future prefetch work should use an async
copy stream plus pinned CPU buffers and should be MTP-aware: MTP verification
does more per-step layer work by validating multiple drafted tokens, giving a
larger compute window in which CPU/NVMe stage-in can overlap with verifier
attention/FFN work.

---

## CLI parameters

```
--kvmem        Master switch (default OFF; off == byte-identical to plain).
--kvmem-block-tokens N      Block granularity in tokens (multiple of KV page size). Default 128.
--kvmem-budget N     Max window tokens kept per selection. Default 131072.
--kvmem-interval N   Decode steps between reselections. Default 64.
--kvmem-sink-blocks N       Always-kept prefix blocks. Default 1.
--kvmem-recent-blocks N     Always-kept suffix blocks (0 = derive from budget). Default 0.
--kvmem-method M     Selection signal: retrieval | h2o | recency. Default retrieval.
--kvmem-select-policy M     Selection policy: topk | quota. Default topk.
--kvmem-retrieval-method M  Retrieval scorer: mean_attention | content_mean. Default mean_attention.
--kvmem-update-mode M       Reselect cadence: interval | step. Default interval.
--kvmem-retrieval-blocks N  Retrieval quota under quota policy (0 = derive). Default 0.
--kvmem-profile-blocks N    Profile quota under quota policy (0 = derive). Default 0.
--kvmem-gpu-memory-ratio F  Fraction of total GPU memory used to estimate KVMem KV cap. Default 0.50.
--kvmem-gpu-high-watermark F  Future spill threshold for GPU-resident blocks. Default 0.95.
--kvmem-gpu-low-watermark F   Future eviction target after spill. Default 0.85.
--kvmem-cpu-bytes N         CPU tier byte budget. 0 disables runtime page release. Default 0.
--kvmem-nvme-dir DIR        NVMe backing-file directory.
--kvmem-nvme-bytes N        NVMe tier byte budget. Requires --kvmem-nvme-dir.
```

`--kvmem-method` picks what ranks the middle (non-sink/recent) blocks each
reselection — all three feed the same `pick_topk_blocks` (sink + recent are
always kept), differing only in what writes `attn_score`:

- `retrieval` (default): global content-frame similarity (§2b). Re-ranks **all**
  historical blocks against the current query and can **resurrect** a dropped
  block. Auto-falls-back to `h2o` when the index/query is not live (q8/fp8 cache,
  or the first post-prefill reselect before any query exists).
- `h2o`: window-local cumulative attention heat only (§2). A heavy-hitter
  **retention** signal — keeps attended blocks resident but cannot resurrect a
  dropped one. Skips the content-index build + per-step query de-RoPE.
- `recency`: no learned signal; `pick_topk_blocks` keeps only the sink + recent
  windows. The cheapest path (no score kernels at all).

`--kvmem-select-policy=quota` separates fixed and scored selection. Sink and
recent blocks are hard-kept first. The remaining budget is filled from the
retrieval score pool and the previous-step profile score pool, then any leftover
capacity is filled by the active method's default score. This is the first
engineering hook for comparing multiple selectors without changing the window
assembly or tiering code.

`--kvmem-retrieval-method=mean_attention` treats the retrieval logits from the
current content-frame query and each block's mean content key as a block-level
attention approximation: it applies a softmax across indexed historical blocks
and uses the resulting probability mass as `retrieval_score`. `content_mean`
keeps the raw dot-product ranking. Both methods score all indexed historical
blocks and can resurrect blocks that are outside the current GPU window.

`--kvmem-update-mode=step` disables decode-interval reselection. The executor
still registers committed decode tokens, still accumulates the step profile, and
still assembles the first post-prefill working set; it does not refresh the
working set again until the next request/step entry point. This preserves MTP
correctness because KVMem metadata is advanced only after verifier-committed
tokens, not speculative draft tokens.

The GPU memory ratio is currently used to estimate a maximum active working-set
budget:

```
estimated_block_bytes =
    block_tokens * standard_attention_layers * n_kv_heads * head_dim
  * 2  // K and V
  * kv_element_bytes

estimated_gpu_blocks =
    floor(total_device_memory * kvmem_gpu_memory_ratio / estimated_block_bytes)
```

When the backend can report total device memory, `select_budget` is clipped to
this estimated block capacity. CPU/NVMe tier slot sizing uses the same block-byte
estimate.

Runtime tier budgets:

```
--kvmem-cpu-bytes N
--kvmem-nvme-dir DIR
--kvmem-nvme-bytes N
```

`--kvmem-cpu-bytes=0` leaves runtime page release disabled: KVMem still runs a
sparse window, but cold blocks remain GPU-resident. Setting a positive CPU
budget enables GPU->CPU stage-out for unselected blocks. If the CPU tier is full
and NVMe is configured, the CPU LRU block is synchronously written to the NVMe
backing file before the slot is reused.

### Tiering implementation status

The current implementation has the following tiering pieces in place:

- `PinnedKvTier` tracks CPU pinned slots, block residency, LRU order, explicit
  LRU eviction, and slot reuse.
- `NvmeKvTier` owns a raw backing file, fixed-size slots, block-to-slot
  residency, synchronous read/write, explicit LRU eviction, and release/reuse.
- `DeviceBackend` exposes raw byte D2H/H2D copy hooks so executor code can move
  KV payloads without depending on the KV dtype.
- `KvMemBlock` carries CPU/NVMe slot handles plus in-flight and dirty flags.
- `QwenExecutor` can copy unselected blocks from GPU KV pages into the CPU tier,
  release their logical pages as non-resident (`page=-1`), spill evicted CPU
  slots to NVMe, and stage selected CPU/NVMe blocks back into newly allocated
  GPU pages before window assembly.
- `QW3_KVMEM_TIER_TRACE=1` emits diagnostic `stage_out`, `cpu_evict`, and
  `stage_in` lines. The model-gated E2E runner uses this to assert that tiered
  sparse runs actually exercise GPU->CPU/NVMe stage-out and CPU/NVMe->GPU
  stage-in.

Two architectural constraints remain:

- Runtime release is opt-in through `--kvmem-cpu-bytes`. The default preserves
  the original sparse-window behavior without releasing GPU pages.
- In the non-continuous single-request path, each standard-attention layer's KV
  tensor is still allocated for the full `ctx_size` physical page capacity.
  Therefore CPU/NVMe stage-out reduces active GPU page residency and enables the
  logical/non-resident state machine, but it does not yet reduce the initial
  CUDA KV tensor allocation. The next step is a bounded KVMem GPU physical page
  pool whose capacity is independent from logical context length.

Internal diagnostic (default off, env-gated): `QW3_KVMEM_TRACE=1`
prints the top attention-heat blocks of each interval.

---

## Validation

Block-sparse has no end-to-end numerical reference (3/4 layers are dense DeltaNet
that see all tokens regardless of selection, and attending over a different block
set is by design a different computation). So the re-RoPE math is proven at the
kernel level, and end-to-end testing proves two properties:

1. **Zero-impact under identity selection.** With a budget large enough to select
   all blocks, every remap skips, the window equals the live cache, and decode is
   byte-identical to plain — confirmed verbatim even with the per-step score
   kernel, kbar rebuild, and drain all running.
2. **Sparsity actually engages.** With a tight budget, the model genuinely cannot
   see dropped blocks (a codeword placed in a dropped middle block is not
   recalled), and the output stays coherent (not garbage) — proving decode runs
   over the reduced re-RoPE'd window and the window is numerically sound.
3. **Selection signal is meaningful.** The internal trace shows per-block scores
   spanning ~10× within an interval, dominated by the attention-sink block and
   the recent/question block (the textbook sink + locality pattern), with the hot
   block's score growing monotonically as decode keeps attending to it — not flat
   noise.

Kernel-level gold standards: `tests/rope_remap_parity.cu`,
`tests/kvmem_enabled_assembly.cu`. Host-logic: `tests/kv_block_store_test.cpp`,
`tests/pinned_kv_tier_test.cpp`. Full `ctest` suite (6 tests) passes.

---

## File map

| Concern | Location |
| --- | --- |
| Block store + selection + remap plan (host) | `include/qw3/kvmem_store.hpp`, `src/kvmem_store.cpp` |
| Window assembly, decode substitution, kbar/score/drain wiring | `src/qwen_executor.cpp`, `src/qwen_executor.hpp` |
| re-RoPE + selection-signal CUDA kernels | `src/kernels_cuda.cu` (`rope_block_remap_paged_kernel`, `block_kmean_paged_kernel`, `block_attn_score_step_kernel`) |
| Global content-frame retrieval kernels | `src/kernels_cuda.cu` (`block_kmean_content_paged_kernel`, `derope_query_kernel`) |
| Global retrieval wiring (build/snapshot/score) | `src/qwen_executor.cpp` (`kvmem_build_content_index`, `kvmem_snapshot_content_query`, `kvmem_retrieval_score`) |
| Backend interface (virtuals + defaults) | `include/qw3/device_backend.hpp` |
| Backend wiring (config, register, reselect cadence) | `src/qwen_native_backend.cpp` |
| CLI flags | `src/qw3_cli.cpp`, `include/qw3/qw3.hpp` |
| CPU/NVMe tier allocators | `include/qw3/pinned_kv_tier.hpp`, `include/qw3/nvme_kv_tier.hpp` |
| Runtime tier stage-in/out + canonicalization | `src/qwen_executor.cpp` (`kvmem_stage_in`, `kvmem_stage_out`, `kvmem_canonicalize_block_for_tier`) |

## Status

Complete: #37 block store, #38 re-RoPE kernel + math, #39 page-index assembly +
byte-lossless, #40 cumulative-attention selection (window-local retention),
#43 CPU pinned tier skeleton, #48/#49 global content-frame KV retrieval
(resurrection of dropped blocks), CPU/NVMe stage-out/stage-in with canonical
tier backing, and model-gated tier E2E coverage. Pending: bounded GPU physical
page pool, prefill-time offload, async copy/NVMe prefetch overlap, true pinned
host buffers, and KVMem-aware continuous+MTP batching.

Known limitations of the current retrieval signal (future work): (1) the mean
key over a 128-token block dilutes a short codeword, so smaller `--kvmem-block-tokens`
sharpens retrieval; (2) the raw `q·k̄` dot is unnormalized, so high-norm sink/early
blocks bias the ranking — a cosine or norm-corrected score would sharpen it; (3)
the index covers prefill blocks only (decode-generated blocks are recency-kept).
