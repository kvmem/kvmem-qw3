# KVMem Implementation Notes for Method Writing

This document summarizes the current KVMem implementation in the `qw3` codebase
as detailed source material for writing an academic method section. It is not
intended to be a concise paper draft. Instead, it records the mechanisms,
design invariants, equations, scheduling decisions, data structures, efficiency
arguments, limitations, and paper-relevant interpretation of the implementation.

> **Active issue tracker:** unresolved correctness bugs, retrieval-quality
> limitations, compatibility gaps, validation evidence, and recommended repair
> order are maintained in `docs/kvmem_known_issues.md`.
>
> **GPU memory optimization record:** the current bounded-page-pool design,
> immutable raw-K layout, FP8/FP16 dtype split, scratch/workspace reductions,
> measured GPU peaks, recommended 48 GiB profiles, and remaining 10M-context
> scaling limits are maintained in `docs/kvmem_gpu_memory_optimization.md`.
>
> **Performance evaluation record:** the controlled 512K ablation methodology,
> exact runtime parameters, stage-in/stage-out/assembly breakdown, GPU-memory
> control, complete implemented-optimization inventory, raw artifact paths,
> and reproduction commands are maintained in
> `docs/kvmem_performance_evaluation_20260726.md`.

The implementation is centered around three ideas:

1. Step-level memory scheduling: KVMem updates its memory state at structured
   execution points, especially after prefill and at configured reselection
   boundaries, rather than continuously recomputing memory decisions for every
   generated token.
2. KV-native memory retrieval: KVMem retrieves historical context blocks using
   model-internal query/key signals, with the default query-conditioned method
   using a mean-key index and a block-level softmax scorer.
3. Tiered KV memory management: KVMem stores the full recoverable KV state across
   a bounded GPU page pool, demand-allocated CPU memory, and NVMe storage, while
   using bounded pinned slabs for transfer and assembling only a selected
   working set for attention.

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
- `include/qw3/pinned_kv_tier.hpp`: CPU-tier metadata, heat-aware/LRU admission,
  and logical slot allocator; immutable mode may back those slots with sparse
  pageable slabs rather than pinning the entire tier.
- `include/qw3/nvme_kv_tier.hpp`: NVMe slot metadata and backing-file I/O.
- `include/qw3/device_backend.hpp`: backend abstraction for page-table operations,
  raw byte copies, KVMem kernels, and the reusable pinned host buffer pool.

> **Experimental method status (2026-07-12):** the DeltaNet state-edit retrieval
> path is retained in the source tree for reproducibility but is not recommended
> and is no longer exposed through the CLI. Its design, measurements, limitations,
> and possible future work are recorded in
> `docs/kvmem_deltanet_retrieval_experimental.md`.

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

After prefill, a newly registered block is canonical:
`baked_pos == orig_pos_start`. When a selected block is placed into a compact
window, its keys may be re-RoPEd in place from `baked_pos` to the new window
position. This makes the selected window stay within the model's normal
positional range. In the current default immutable mode, position-free raw K
in CPU memory is authoritative, active GPU K is a disposable materialization,
and ordinary CPU/NVMe block records contain V rather than another K copy. The
legacy mutable-K path instead re-canonicalizes K before spilling it.

This gives the implementation an important invariant:

> GPU-resident working K may be baked either at original positions or at current
> window positions, and `baked_pos` records which. Immutable mode reconstructs
> K from position-free CPU raw K and restores V from CPU/NVMe; the legacy mode
> stores spilled K in canonical original-position form.

This invariant makes later stage-in and remap safe: a recalled immutable block
is baked exactly once from raw K at its target slot, while a still-resident block
may use bounded in-place delta rotation before the refresh threshold forces a
raw rebuild.

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
- `sink_blocks` / `recent_blocks`: resolved physical prefix/suffix bands. By
  default they are derived after `block_tokens` is known from token targets:
  sink=`clamp(1% × budget, 1K, 2K)` and
  recent=`clamp(8% × budget, 4K, 16K)`, each rounded up to complete blocks.
  Explicit `--kvmem-{sink,recent}-tokens` overrides the corresponding automatic
  target; the legacy block flags remain mutually-exclusive compatibility
  overrides. Explicit `--kvmem-recent-blocks 0` means no suffix blocks are kept
  unconditionally.
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
   - If a new turn resumes an already sparse session, the inherited semantic
     window is normalized through `kvmem_prepare_prefill_window` before the
     first new token is evaluated.
2. During prefill:
   - Newly written tokens are registered into `KvMemStore` through
     `kvmem_register_until` / `kvmem_register_append`.
   - Query rows inside the marked question span are captured and de-RoPEd.
   - Per-block content mean keys are captured from freshly produced K batches.
   - If a bounded GPU pool is close to exhaustion, KVMem triggers a deterministic
     in-prefill pressure selection/offload through
     `kvmem_maybe_prefill_offload`.
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
small cushion of recent blocks. If not, it calls
`kvmem_reselect_prefill_pressure` rather than the semantic retrieval selector.

The pressure policy is deterministic. It keeps the configured sink prefix and
fills every remaining budget slot with the newest historical blocks. The word
"recent" in this policy means the entire newest tail left after reserving the
sink; it is independent of the smaller `recent_blocks` region optionally pinned
by the semantic selector. Attention, retrieval, and profile scores are ignored.

This separation matters for multi-turn execution. A restored checkpoint carries
the previous turn's assembled semantic window. If the next turn starts a long
prefill directly on that window, some new hidden states are shaped by the old
query before page pressure first triggers. `kvmem_prepare_prefill_window`
therefore normalizes every already-sparse continuation to the same sink-plus-tail
policy before its first suffix token. P and M checkpoint reuse and persistent
`reset_session=false` growth follow the same rule. The post-prefill selector is
still query-conditioned mean-k/H2O/recency and is not changed by this mechanism.

This avoids a failure mode where the next chunk allocates pages in one burst and
exhausts the pool before the scheduler has a chance to evict. It also spreads
tiering cost over prefill rather than concentrating all eviction at the
prefill-to-decode boundary.

For method writing, this is the key argument:

> KVMem schedules memory at semantic step boundaries but also includes a
> resource-triggered, query-independent sink-plus-tail offload path to maintain
> the bounded-GPU invariant during very long prompt ingestion. Semantic
> retrieval is reserved for the prefill-to-decode boundary.

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

Production mean-key storage is IEEE FP16 (`__half`, not BF16), independently
of whether active K/V uses FP16 or FP8 E4M3. FP32 KV keeps a diagnostic FP32
index mode. Builders, dot products, softmax statistics, and score accumulation
remain FP32. FP16 halves index bytes relative to FP32 without quantizing the
already averaged vector as aggressively as the reverted FP8-index experiment.

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

#### 4.4.1 Message-role policy and planned API metadata (2026-07-15)

The current server does not receive an explicit semantic distinction between
retrieval corpus and retrieval query. Its automatic policy is to use the content
span of the last ordinary `user` message (excluding a tool-response wrapper) as
the query. This is a useful default for chat and agent traffic, where a new step
normally ends with a user task, but it is not a general context detector. In
particular, if a request ends with one very large user message containing only
history, that whole message is considered the query and the executor's 512-token
cap keeps only its first 512 query tokens.

The agreed first-stage policy is therefore:

- the last ordinary `user` message is the retrieval query;
- earlier `user` messages, prior `assistant` messages, and tool results are the
  retrieval corpus;
- genuine `system` / `developer` instructions are control context and should
  eventually be semantically pinned rather than merely relying on the fixed
  prefix sink;
- the current query and the live recent agent/tool trajectory should remain in
  the active window.

Until explicit API metadata is implemented, every query-conditioned experiment
must end with a separate, short user query. A context warm-up must not end with
the long context message itself. Append a user message such as:

```text
Please remember the information above and reply with a brief confirmation.
```

That short message is intentionally the warm-up query. It prevents the long
context from being interpreted as the query and, under the current
implementation, activates incremental mean-key capture for the complete prompt.
The later real request appends a new final user message, which then becomes the
new retrieval query. The confirmation query can influence the warm-up working-set
selection; this is a documented property of the temporary experimental protocol,
not a claim of equivalence to a future context-only prefill mode.

The planned OpenAI-compatible API extension is a top-level, namespaced object:

```json
{
  "kvmem": {
    "mode": "last-user | explicit | context-only",
    "pinned_message_indices": [0],
    "context_message_indices": [1],
    "query_message_indices": [2]
  }
}
```

The modes are intended to mean:

- `last-user`: the automatic policy above and the initial implementation target;
- `explicit`: the caller supplies message indices for pinned control text,
  retrieval corpus, and query;
- `context-only`: build the complete context index during a prefill-only warm-up
  without inventing a retrieval query.

The explicit and context-only modes are a recorded design, not current API
behavior. Their implementation also requires context mean-key construction to be
independent of query availability: context keys depend only on the context, while
query capture and retrieval scoring depend on the later query. Message rendering
should eventually return token spans per message, blocks containing pinned/query
tokens should be pinned as whole blocks, and only corpus blocks should compete in
retrieval top-k.

#### 4.4.2 Warm-prefix checkpoint and query-replay invariant

For an above-budget query-conditioned request, prefix reuse and query replay
share one correctness boundary. Let:

- `D` be the longest common token prefix,
- `C` be the restored warm checkpoint,
- `Qb` be the query start,
- `B = align_down(Qb, block_tokens)` be the replay boundary.

The backend only accepts P/M checkpoints satisfying:

```text
C <= min(D, B)
```

If no cached checkpoint satisfies the inequality, the request cold-prefills.
Restoring a checkpoint after B is not recoverable: it omits query rows before C
and there is no inverse operation that reconstructs the recurrent/KV state at B.

A valid warm hit follows the same replay transaction as a cold request:

```text
restore C -> prefill C..B -> snapshot B -> prefill B..end
-> semantic selection -> restore B while retaining selected context
-> replay B..end -> decode
```

The resumable prompt checkpoint P is a second prefill boundary. When `P <= B`,
it is captured on the first pass. When `P > B`, it is captured during the final
selected-context replay, because the first-pass state at P is deliberately
discarded. Both plain and MTP paths enforce complete query capture before
selection and again after replay.

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

The sink region preserves stable prefix information. A configured recent region
prevents the model from losing the immediate local continuation. A value of zero
disables unconditional suffix retention; retrieval may still select tail blocks
normally when their scores rank into the budget.

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

#### 6.4.1 Immutable raw-K construction

The default path above changes fp16 K in place.  In a normal single reselect
this adds only one rounding step, but transcript replay can move the same block
thousands of times.  Each inverse/forward RoPE then starts from the previously
rounded result, so the numerical error is cumulative and affects every later
prefill/decode attention that reads the block, not only retrieval scoring.

The drift-bounded construction path is enabled by default. `--kvmem-immutable-k`
can state that choice explicitly, while `--no-kvmem-immutable-k` selects the
legacy in-place ablation. `QW3_KVMEM_IMMUTABLE_SOURCE_K=0|1` remains as a
backward-compatible override for existing scripts:

1. Every standard-attention K row is captured **before RoPE**, converted to the
   configured KV dtype, and stored in demand-allocated ordinary-memory CPU
   chunks indexed by `(layer, true token position)`. The default chunk spans
   2048 tokens (`QW3_KVMEM_RAW_K_CHUNK_TOKENS` can override it); `--ctx` is a
   logical capacity and no longer causes a dense raw-K allocation at startup.
2. GPU holds only one active K and one V copy. `baked_pos`, `remap_count`, and
   cumulative absolute position displacement describe that active K.
3. Small moves use the existing batched in-place delta re-RoPE kernel.
4. A cold stage-in, 32 accumulated moves, or 262144 accumulated token positions
   triggers a raw refresh. CPU gathers selected raw blocks into a contiguous
   pinned buffer, performs a large H2D, and a CUDA kernel scatters each block to
   its physical pages while applying destination RoPE exactly once.
5. Standard-layer spill records contain V only; K is restored from the raw
   mirror. When MTP local-position mode is active, the MTP layer follows the
   same ownership split: raw MTP K is stored by true logical token position and
   its tier record contains V only.

For Qwen3.6-27B at a 256K fp16 active capacity this removes the previous
approximately 8 GiB GPU working-K mirror. Persistent raw K grows by
approximately 8.5 GiB per 256K true-context tokens when the 16 standard layers
and local-position MTP K are both enabled. Raw K and immutable-mode CPU spill
blocks share one strict `--kvmem-cpu-gb` budget. The CPU spill cache allocates
pageable backing per occupied slot; when a new raw-K chunk would exceed the
budget, CPU spill LRU blocks are demoted to the disk tier first. This avoids
both the old dense `--ctx` raw-K reservation and a second, fully pinned CPU
pool.

The configure-time capacity check remains conservative: it reserves the
maximum padded raw-K size when proving that CPU plus disk can hold a prompt
that reaches `--ctx`. Therefore demand allocation changes physical memory use,
not the safety guarantee. Transient capture and gather buffers remain bounded
(about 64 MiB and 128 MiB respectively with the default 2048-token chunk and
128-block transfer batch).

A 2026-07-23 smoke with `--ctx 2000000`, fp16 KV, 72 GiB CPU, 224K selection,
32K generation reserve, and MTP-4 observed:

- startup: `raw_k_allocated_bytes=0`, no large pinned CPU spill slab;
- 63,019 prompt tokens: 2.06 GiB raw K, zero spill and zero disk;
- 280,031 prompt tokens: 9.10 GiB raw K, 1.64 GiB CPU spill, zero disk;
- the latter completed pressure stage-out, 896-block stage-in, and query replay.

Correctness guards/tests:

- `KvMemStore` tests cover resident delta moves, threshold refresh, and mandatory
  cold stage-in refresh. They also verify that mandatory query-replay blocks
  replace ordinary top-k candidates instead of expanding the selected window
  beyond its token budget.
- `qw3-kvmem-immutable-k` verifies raw scatter+RoPE is byte-identical to a
  direct one-shot fp16 bake and retains the 1800-remap drift control.
- The multi-turn prefill-pressure smoke covers tiering, prefix reuse,
  query-conditioned re-selection, query replay, and MTP.

The mode requires fp16, fp32, or fp8 (q8 row-scale
re-RoPE remains unsupported) and a bounded tiered GPU pool. FP8 halves both the
active KV and CPU raw-K bytes relative to fp16, at the cost of one-time
quantization error.

#### 6.4.2 Planned I/O–Compute Overlap for Raw-K Materialization

The bulk raw-K path removes the launch and DMA setup overhead of thousands of
small page copies, but its first implementation is intentionally conservative:

```text
CPU gather -> H2D and wait -> GPU scatter+RoPE -> attention
```

The next optimization should turn this into an asynchronous pipeline. This is
an execution optimization, not a change to retrieval or immutable-K semantics.

##### Dependency boundary

A layer may execute attention only after that layer's selected K/V pages are
ready. It does **not** need to wait for every later layer. Similarly, an evicted
page may be reused only after its final reader and its D2H transfer have both
completed. These are the two ordering constraints the pipeline must preserve.

The following operations can overlap:

- CPU gathering batch `N+1`;
- H2D of packed batch `N`;
- GPU scatter+RoPE of batch `N-1`;
- Transformer computation that does not read the pages being written;
- asynchronous stage-out of blocks whose last attention reader has completed;
- speculative CPU/HDD prefetch of likely next-round candidates.

The same block/layer cannot be scattered while attention reads it, and the
first attention layer's compulsory load remains on the critical path.

##### Double-buffered transfer pipeline

Use two pinned host buffers and two device staging buffers:

```text
time 0: CPU gather H0
time 1: H2D H0->D0              || CPU gather H1
time 2: scatter+RoPE D0         || H2D H1->D1 || CPU gather next H0
time 3: scatter+RoPE D1         || H2D H0->D0 || CPU gather next H1
```

The H2D copy stream records a completion event for each device buffer. The
materialization stream waits on that event before launching scatter+RoPE. A
buffer may be reused only after both its DMA and scatter event have completed.
H2D and scatter both consume HBM bandwidth, so perfect overlap is not assumed;
the primary goal is to hide CPU gather and DMA setup latency.

##### Layer-wise query-replay pipeline

The strongest compute overlap is to materialize K/V by Transformer layer and
pipeline it with the mandatory query replay:

```text
load layer 0 KV
replay/query compute layer 0 || load layer 1 KV
replay/query compute layer 1 || load layer 2 KV
...
replay/query compute layer L || load layer L+1 KV
```

Each layer gets a `kv_ready_event[layer]`. Before its attention operation, the
execution stream waits only for that layer. Projection, RMSNorm, DeltaNet, and
FFN work from earlier layers can therefore hide later-layer PCIe transfers.
Only layer 0's load and any unhidden pipeline tail contribute directly to
visible reselection latency.

This requires changing the current all-layer packed materialization into
layer-major transfer batches. The CPU raw-K mirror is already layer-major, so
this does not require changing its persistent layout.

##### V and stage-out

After raw-K batching, cold-block V page copies are likely to become the next
bottleneck. V should use the same large-transfer pattern:

```text
CPU gather selected V -> bulk H2D -> GPU V scatter
```

K and V for a layer may share one packed transfer or use separate buffers,
depending on whether fused scatter improves memory coalescing.

Stage-out should become non-blocking. GPU pages transition through:

```text
resident -> evicting -> free
```

The D2H stream waits on the block's last-reader event. The page allocator cannot
reissue an `evicting` page until the D2H completion event fires. A small pool of
spare pages is needed so subsequent prefill can continue while eviction drains.

##### Reducing I/O before hiding it

Overlap complements, but does not replace, reducing transferred bytes:

- transfer only `new_selection - old_selection`;
- keep a small GPU victim cache above the strict attention budget;
- add selection hysteresis to avoid blocks oscillating around the top-k cutoff;
- apply raw refresh only to cold, over-rotated, or out-of-range blocks;
- merge adjacent source blocks into one CPU copy or disk extent;
- keep high-score/recent V blocks in DRAM and use HDD only for cold data;
- optionally transfer raw K/V in FP8 after accuracy validation.

For HDD-backed blocks, random seek latency dominates. Reads should be sorted by
block ID, adjacent ranges coalesced, and a cheap top-M candidate pass used to
start speculative prefetch before final top-k selection when possible.

##### Recommended implementation order

1. Remove per-block transfer waits and introduce CUDA events.
2. Add double-buffered CPU gather/H2D/scatter for raw K.
3. Convert raw-K materialization to layer-wise operation.
4. Pipeline layer-wise loading with query replay.
5. Batch V H2D and add a V scatter kernel.
6. Add asynchronous `evicting` page state for stage-out.
7. Add selection hysteresis, victim-cache residency, and HDD top-M prefetch.

Required evaluation should report CPU gather, H2D, scatter+RoPE, stage-out, and
visible reselection time separately, plus overlap efficiency, transferred
bytes, cold-block ratio, TTFT, TPOT, throughput, and peak GPU/CPU memory.

#### 6.4.3 Measured Reselection Profile (2026-07-23)

`QW3_KVMEM_PERF_TRACE=1` emits one `[kvmem-reselect-perf]` row per
explicit pressure selection and semantic selection. It measures selection,
stage-out subphases, asynchronous-read duration and actual wait, H2D
enqueue/wait, the prepare/finish overlap gap, assemble, bytes, and block
counts. The flag synchronizes timed GPU regions and is diagnostic, not a
throughput-benchmark setting.

The first measured LongMemEval-M sample (`19b5f2b3`) used 1,119,177 prompt
tokens, 224K context budget, 32K generation reserve, block size 32, FP16 KV,
immutable K, query replay, MTP-4, 72 GiB CPU budget, and HDD-backed storage:

| Semantic reselection component | Measured time |
|---|---:|
| mean-K scoring and selection | 1.001 s |
| stage-out, 5,826 blocks / 6.045 GiB | 6.264 s |
| construct per-block HDD read buffers | 3.424 s |
| HDD read, 5,572 blocks / 5.781 GiB | 78.814 s |
| HDD-buffer-to-GPU enqueue | 1.071 s |
| assemble / raw-K materialization / re-RoPE | 3.620 s |
| semantic reselection total | 94.686 s |

The prepare/finish overlap gap was only 0.009 ms and the caller waited
78.814 s for a 78.814 s HDD read. The existing async future therefore provides
asynchrony in the API but hides essentially none of this request's I/O. HDD
throughput was about 75 MiB/s. A 6-GiB direct sequential read of the same
backing file took 37.42 s (about 164 MiB/s). If physical-slot sorting and
coalescing approach that sequential rate, the measured 5.781-GiB read would
fall from 78.8 s to roughly 36 s; this roughly 43-s difference is an upper
bound, not an expected result. The matching selection contains about 2,030
contiguous cold-block runs (mean run 2.86 blocks, median gap 10 blocks), so
many physical seeks remain even after adjacent blocks are merged. Reading
through only one- or two-slot gaps and measuring several extent thresholds is
needed to find the real optimum.

There were also 31 prefill-pressure selections. Before the pinned V tier
filled, each 896-block / 0.930-GiB event took about 0.33 s. Once it filled,
synchronous HDD victim writes raised an event to about 0.9--1.2 s. Cumulative
request timing was 1.001 s retrieval, 83.801 s stage-in, 27.870 s stage-out,
and 9.367 s assemble. TTFT was 761.9 s, so mean-K scoring itself was not the
throughput bottleneck; storage movement was about 16% of TTFT.

The measured optimization order is therefore:

1. Replace thousands of per-block `std::vector` allocations with a fixed
   double/triple-buffered pinned slab pipeline.
2. Sort HDD reads by physical slot offset and coalesce adjacent slots. The
   current 5,572 `fseek`/`fread` operations are the dominant latency.
3. Stream `HDD read -> pinned slab -> H2D` so the final 1.071 s H2D phase is
   hidden under reads instead of beginning after all reads finish.
4. Make pressure stage-out asynchronous with `resident -> evicting -> free`
   pages; the 28.7K-token gap to the next pressure event is large enough to
   hide its roughly one-second spill.
5. Track an inclusive clean CPU/HDD backing copy for immutable historical
   blocks. A clean GPU block can then be dropped without another D2H/write,
   avoiding repeated exclusive-tier traffic.
6. Only after these changes, pipeline layer-wise materialization with query
   replay. This is more invasive and cannot hide the compulsory layer-0 load.

Starting HDD reads before semantic stage-out can hide at most the 6.264-s
stage-out phase, and concurrent reads/writes on the same mechanical disk may
instead increase seek cost. It is lower priority than coalescing, streaming,
and eliminating redundant writes.

#### 6.4.4 Storage and Prefetch Plan for 10M Context

The production target is PCIe NVMe SSD, not the rotating `/data` device used by
the 2026-07-23 diagnostic. The complete SSD-oriented storage layout, io_uring
engine, registered slab pool, bulk H2D/scatter pipeline, cache policy, and
implementation order are specified in
`docs/kvmem_ssd_complete_design.md`. The earlier
`docs/kvmem_nvme_ssd_architecture.md` is retained as preliminary design and
profiling history.

At 10M fp16 tokens, immutable raw K alone is approximately 324 GiB and
historical V is of similar order. A 72 GiB CPU budget cannot keep raw K
authoritative in DRAM, so demand allocation by itself is not sufficient. The
next storage format must make raw-K chunks tierable:

```text
raw-K chunk: CPU-resident | disk-resident | prefetching
V block:     GPU-resident | CPU-resident | disk-resident | prefetching
```

Cold raw-K chunks and V blocks should use one block-addressable disk repository
with an in-memory location table. Selected raw K and V can then be fetched by
the same extent planner. CPU DRAM becomes a cache, not a requirement that scales
with logical context length.

The SSD prefetch pipeline should be:

1. Submit exact selected reads immediately after scoring, before synchronous
   stage-out, because host slabs do not require free GPU pages.
2. Use io_uring fixed-file/fixed-buffer batches at a benchmarked queue depth
   (initially 32), rather than a single sequential reader.
3. Read independent 1--8 MiB extents concurrently; coalesce truly adjacent
   records to reduce submission overhead, but do not read unused gaps merely to
   avoid HDD seeks.
4. Read extents into two or three fixed pinned slabs. Do not allocate one
   `std::vector` per block.
5. While slab N+1 is being read, issue one bulk H2D for slab N into a reusable
   device staging buffer and scatter its K/V pages with a CUDA kernel.
6. Optionally prefetch a conservative top-M superset from mandatory blocks,
   previous-turn winners, or partial-query scores; keep useful extras in a
   bounded CPU/GPU victim cache.
7. Record `candidate_hit_rate`, extra bytes read, extent count, queue depth,
   disk read time, hidden read time, H2D time, and visible wait.

The old HDD result remains useful as evidence that the current `FILE*` interface
does not overlap I/O, but its single-reader and seek-avoidance recommendations
are not the target architecture.

### 6.5 MTP Window Mirror

When MTP speculative decoding is active and KVMem tiering is enabled for the MTP
KV cache, the executor builds a separate MTP window page table over the MTP
cache. Long sessions use two independent coordinates:

- **logical position**: monotonic token/page identity, allowed to exceed
  `n_ctx_train`;
- **RoPE position**: the token's slot in the current compact selected window,
  required to remain below `n_ctx_train`.

With immutable K, `QW3_KVMEM_MTP_LOCAL_POSITIONS=1` is the default. MTP prefix
priming receives both coordinates. It appends K/V through the logical MTP page
table, but rotates Q/K using the exact compact position used by the target-model
chunk. Before RoPE, projected MTP K is converted to the configured KV dtype and
captured in a CPU raw-K mirror indexed by logical token position.

At re-selection, selected MTP K is never recovered by inverting a previous
rotation. The executor gathers the selected raw rows, transfers them in bounded
block batches, scatters them to the selected MTP physical pages, and applies
RoPE once at each `to_base`. MTP V remains position-independent and is loaded
from the ordinary CPU/NVMe tier. `mtp_baked_pos_` remains for diagnostics and
the legacy `QW3_KVMEM_MTP_LOCAL_POSITIONS=0` A/B path.

Pressure stage-out is deferred until the target chunk's MTP prefix has been
primed. This ordering is required: otherwise the last block of a chunk can be
evicted after target K/V append but before its MTP K/V exists, producing an
empty MTP tail in the tier record.

Decode-time re-selection can start CPU/NVMe prefetch before the just-accepted
target rows have finished MTP prefix reconstruction. In local-position mode,
historical MTP V stage-in is therefore independent of the transient
`mtp_prefix_len >= registered_pos` predicate: the decision is frozen when the
prefetch starts, V transfer overlaps prefix reconstruction, and assembly waits
for both. The legacy MTP path retains the coverage gate because its absent
sparse page table must not be allocated as a dense logical prefix.

The executor admits local-position MTP only when
`select_budget + gen_budget <= n_ctx_train`; the backend also checks the
request's `select_budget + max_tokens`. Mandatory query-replay suffix blocks
consume slots inside the top-k budget rather than being appended afterward, so
the assembled window cannot silently exceed `select_budget`. Draft depth is
additionally clamped at runtime against the remaining compact positions. Target
prefill/verify/decode, every MTP prefix rebuild, and raw-K materialization all
have executor-side range checks. Position modulo or clamping of an individual
logical position is intentionally not used.

## 7. Tiered KV Memory Management

### 7.1 Problem

A long session can contain far more KV than fits on GPU. Even if full KV could
fit, attending over all of it would make decode cost grow with total history.
KVMem needs to keep only a bounded working set GPU-resident while preserving
the ability to recall older KV blocks.

The current implementation uses three tiers:

1. GPU bounded page pool.
2. CPU memory (demand-allocated pageable backing in immutable mode; the legacy
   mutable-K path retains a fixed pinned slab).
3. NVMe backing file.

### 7.2 Block Byte Layout

One tier slot stores one block's tier-owned bytes across all
standard-attention layers. The layout written by
`kvmem_copy_block_to_host_ptr` is conditional:

```text
for each standard attention layer:
  for each page in the block:
    K page bytes                 # legacy mutable-K mode only
    V page bytes
optional MTP segment:
  for each page:
    MTP K page bytes             # legacy MTP-position mode only
    MTP V page bytes
```

Immutable raw-K mirrors are separately charged to the total
`--kvmem-cpu-gb` budget. Consequently the default immutable/local-position
record is standard-layer V plus MTP V; neither K copy is duplicated in the
pinned/NVMe tier.

The legacy estimated block byte size is computed from:

```text
block_tokens * n_kv_heads * head_dim * num_layers * 2(K,V) * elem_bytes
```

For q8 KV, row scale overhead is included in the estimate. Immutable raw-K
currently supports fp16, fp32, and fp8; q8 row-scale raw materialization is
rejected at configuration time.

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

### 7.4 CPU Tier

The metadata class `PinnedKvTier` tracks:

- `slot_count`,
- free slot list,
- `block_id -> slot`,
- LRU order.

The class does not own backing memory and does not issue copies. This separation
keeps tier placement logic testable without CUDA.

In immutable mode, occupied slots receive pageable backing on demand and share
the configured CPU byte budget with demand-allocated raw-K chunks. Releasing or
demoting a slot frees its backing. This avoids reserving or pinning tens of GiB
for unused slots. The legacy mutable-K path still owns or borrows one fixed
pinned `HostBuffer`.

Pinned memory is important because D2H/H2D raw byte copies can be asynchronous
on a copy stream. A pageable buffer would force the driver to serialize through
an internal bounce buffer, increasing stage-out and stage-in cost.

For continuous batching on the legacy path, a `HostTierBufferPool` can recycle
pinned buffers keyed by exact byte size. The optimized immutable transfer path
should instead use a small bounded pinned slab pool between pageable CPU/disk
storage and the GPU.

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
   - in legacy mutable-K mode, re-RoPE main K back to the original position;
   - in legacy MTP-position mode, canonicalize MTP K using its own
     baked-position tracking;
   - immutable raw K requires no canonicalization and is not part of the spill.
5. Begin a device-to-host transfer on the KV copy stream.
6. Copy each tier-owned page into the pinned staging buffer (V-only for the
   default immutable standard/MTP path).
7. Wait for the copy stream.
8. Place the block into the CPU tier if the shared raw-K/spill budget has room.
9. If raw-K growth later needs that memory, demote CPU LRU spill blocks to disk
   before allocating the new raw-K chunk.
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
  - queues H2D copies from its pageable backing (the planned slab pipeline will
    gather these into bounded pinned buffers first);
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
6. Logical page identity and RoPE coordinates are independent; MTP RoPE
   coordinates must remain below `n_ctx_train`.
7. Immutable standard/MTP K authority is unrotated CPU raw K; active GPU K is a
   disposable materialization in the current compact frame.
8. Legacy CPU/NVMe K is canonicalized to original positions; immutable tier
   records contain V only.
9. V does not require re-RoPE; only K is position-rotated.
10. Pressure stage-out must wait until the same target chunk's MTP prefix has
    been captured, then stage-out must precede stage-in for bounded pools.
11. Mandatory query/suffix blocks consume selection-budget slots; they never
    expand the compact attention window beyond `select_budget`.
12. Local-position MTP stage-in restores historical V even while the newest
    accepted prefix row is temporarily catching up.
13. Retrieval index readiness gates query-conditioned scoring; unavailable
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

The production query-conditioned mean-K index is IEEE FP16, independently of
whether active K/V uses FP16 or FP8 E4M3. Builders, dot products, softmax
statistics, and score accumulation remain FP32; FP32 index storage is retained
only as a diagnostic full-precision mode. An earlier FP8-index experiment was
reverted because quantizing an already averaged mean vector changed retrieval
rankings. q8 KV likewise uses the FP16 mean index, while q8 re-RoPE remains
unsupported because its row-scale format cannot use the current standalone
raw-K/remap representation.

### 12.3 NVMe I/O Backend Boundary

The low-level `NvmeKvTier` API uses buffered positional `pread`/`pwrite`.
Adjacent file and buffer spans are coalesced, while upper-level background
workers overlap those blocking syscalls with prefill, CPU admission, and raw-K
assembly. Consequently low-level I/O is synchronous per worker, but proactive
stage-out is not normally synchronous on the model-compute critical path.
`sync_file_range`/`POSIX_FADV_DONTNEED` bound kernel page-cache duplication.
`io_uring`, `O_DIRECT`, and GPUDirect Storage are not implemented yet.

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

### 12.7 LongMemEval-M Diagnosis and Scalable Mean-K Scoring

The original fused mean-k softmax scorer stored every block/sub-block logit in
one CTA's dynamic shared memory. It therefore accepted at most 8192 pages and
could not preserve the same scoring semantics on million-token LongMemEval-M
prompts with 32-token blocks. The fused kernel remains the fast path at or below
8192 pages. Larger inputs now use an exact tiled one-dot implementation:

1. compute and store each tiled \(Q\cdot\bar K\) logit once in bounded device
   workspace;
2. reduce the stored logits to the global max/sum state for each
   layer/query/head distribution;
3. normalize the same stored logits and accumulate block mass.

The workspace scales with one bounded query/distribution batch rather than the
full request's score matrix. The former exact tiled two-dot implementation
(tile LSE, global LSE merge, then dot recomputation) remains available as an
explicit A/B baseline. CUDA parity tests cover the 8192/8193 boundary,
kept-band masking, sub-block reductions, and comparison with the fused kernel,
the two-dot baseline, and a host reference.

A 2026-07-14 diagnostic used 10 LongMemEval-M samples with a 2M context limit,
200K selected-token budget, 32-token blocks, 8 sink blocks, query-conditioned
mean-k, and no sub-block mode. Manual accuracy was 6/10 after enabling the
scalable scorer. Score dumps for the four remaining failures showed that their
gold sessions were not wholly absent. Approximate gold-session block coverage
was 60/194, 30/239, 61/256, and 59/233. Each gold session also contained a
high-ranked block (best ranks 4--402), but only a scattered fraction of the
session survived selection.

The observed failure mode is therefore usually not complete retrieval failure.
It is fragmented evidence retrieval and downstream disambiguation/reasoning:

- a relevant numeric fact can be combined with a similarly worded distractor;
- old and updated facts can both be present without the newest value winning;
- event content can survive while its session date/header does not;
- multiple retrieved events can still be placed in the wrong temporal order.

This diagnosis motivates neighbor expansion and session/header-aware selection
before simply increasing the token budget. It also motivates an oracle test that
forces the answer-local blocks to separate retrieval errors from model reasoning
errors.

#### 12.7.1 Bounded fresh construction for hybrid state consistency

The subsequent frozen-ten repair study found a second, independent failure
mode. Reassembling normal-attention K/V does not reassemble Qwen3.6's 48
DeltaNet layers, and mixing freshly prefetched evidence with a large set of old
locally-constructed K/V produces a window whose hidden representations do not
describe one coherent history.

The current experimental repair keeps the historical repository and mean-K
index, but compiles a small answer window at query time:

1. score the complete historical repository with the ordinary query-conditioned
   mean-K scorer;
2. take 1024 full 32-token source blocks (32K tokens) and restore source order;
3. prefill those exact source tokens once from a clean recurrent state;
4. assemble only the stable eight-block sink plus the fresh blocks before the
   exact final query.

The last step is important. A knowledge-update sample had the new and old facts
at mean-K ranks 3 and 29. Dense replay of the top blocks answered correctly at
32K, 8K, 2K, and 1K, while adding 32K of old cached evidence to the 32K fresh
window reverted to the old fact. Adding roughly 192K of old cached evidence was
worse. Consequently block-refresh mode defaults its cached cap to the configured
sink size; an explicit `QW3_KVMEM_TRANSCRIPT_REFRESH_CACHED_TOKENS` remains only
for controlled A/Bs.

This is not a full-history recomputation: a roughly 1.1M-token transcript still
recomputes only 32K selected source tokens. The frozen-ten run is still in
progress, so this path remains environment-gated and is not yet the production
default.

The same diagnostic exposed a configuration ambiguity: `recent_blocks == 0`
previously derived an unconditional suffix of `budget/4`. At a 200K budget and
32-token blocks this silently pinned 1600 blocks (51,200 tokens). Zero is now
literal and reserves no suffix blocks. Experiments that require recency must set
an explicit nonzero value so their budget allocation is reproducible.

### 12.8 Experimental Clean-Query Capture

An opt-in diagnostic path, enabled only by `QW3_KVMEM_CLEAN_QUERY=1`, prefills
the final question once in isolation and stashes its content-frame query rows.
The normal context prefill then uses those rows for selection before prefilling
the question into the selected decode window. This tests whether query vectors
formed after an already-compressed long prefill are themselves biased toward the
temporary recent window. The feature is disabled by default and does not engage
for warm reuse, prefix-cache capture, or logit-dump requests. It remains an
experimental probe rather than a recommended evaluation default.

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
