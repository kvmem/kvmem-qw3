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
>
> **Adaptive retrieval-index record:** Section 4.9 defines the retrieval
> semantics of direction-adaptive multi-prototype indexing. CPU/GPU index
> placement, one-pass scoring, incremental GPU upload, controlled 1M
> measurements, and implementation-specific performance tuning are maintained
> separately in `docs/kvmem_adaptive_index_optimization_20260730.md`; the
> preceding placement baseline is preserved in
> `docs/kvmem_adaptive_index_placement_ab_20260730.md`.

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
- `src/kernels_cuda.cu`: CUDA kernels for mean-key and adaptive directional
  prototype construction, prototype/page softmax retrieval scoring, exact-mass
  scoring, de-RoPE, and batched re-RoPE.
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
- `retrieval_method`: `mean-k`, `per-token`, `sub-block-mean-k`,
  `key-direction-fixed4`, or `key-direction-adaptive` at the public
  configuration surface. The two Key-direction modes use the SubBlockMeanK
  scorer family with a distinct `prototype_mode`.
- `n_subblocks` and `subblock_reduce`: contiguous sub-block count and
  sum/MaxSim reduction for SubBlockMeanK.
- `adaptive_gain_1to2` and `adaptive_gain_2to4`: relative residual-gain
  thresholds for Adaptive prototype counts.
- `index_placement` and `adaptive_score_mode`: GPU/CPU packed-index placement
  and layer-one-pass/tiled scoring policy. These change capacity and execution,
  not retrieval semantics.
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
   - The executor allocates query capture and Mean-K or Adaptive index state for
     the expected prompt length.
   - If a new turn resumes an already sparse session, the inherited semantic
     window is normalized through `kvmem_prepare_prefill_window` before the
     first new token is evaluated.
2. During prefill:
   - Newly written tokens are registered into `KvMemStore` through
     `kvmem_register_until` / `kvmem_register_append`.
   - Query rows inside the marked question span are captured and de-RoPEd.
   - Content-frame means or Adaptive directional prototype candidates are
     captured from freshly produced K batches.
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

> By default, KVMem schedules semantic memory at request/query boundaries and
> uses a resource-triggered, query-independent sink-plus-tail path during long
> prefill. The opt-in `semantic_chunk` construction experiment instead captures
> a query from every post-threshold physical chunk, selects historical context,
> rolls back, and commits that chunk once under the selected window. Its
> provisional pass has no durable KV/index/tier or MTP side effects.

## 4. KV-Native Mean-Key and Adaptive Prototype Retrieval

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

The current implementation captures the complete marked query span. On the
default CUDA Mean-K/Adaptive path, query rows are stored in FP16. Queries that
fit the bounded scoring stage remain on GPU; a longer span is kept in pageable
host memory and scored in bounded token chunks (256 tokens by default). Chunk
scores accumulate into the same block-score buffer, so query streaming changes
capacity and transfer scheduling rather than retrieval semantics.

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

The implementation now also exposes a middle point between one mean per
32-token slice and full per-token ExactMass:
`--kvmem-retrieval-method key-direction-adaptive`. It retains one, two, or four
directional prototypes per slice and is detailed in Section 4.9. The per-token
exact-mass method remains the stronger but heavier alternative.

For paper writing, it is worth distinguishing:

- KVMem's storage and window assembly machinery can be lossless under an
  all-block budget.
- Retrieval quality depends on the scoring approximation and budget.

This distinction is supported by the current evaluation notes in
`docs/kvmem_utility_eval_plan.md`: all-block KVMem matches full-context behavior
much more closely, while aggressive budget cuts expose retrieval recall limits.

### 4.9 Adaptive Key-Direction Semantic Segmentation

#### 4.9.1 Purpose and exact scope

The contiguous sub-block Mean-K path reduces mean dilution by representing a
large physical retrieval block with one mean for each contiguous 32-token
slice. It still compresses every such slice to a single direction. If the slice
contains several unrelated Key directions, its arithmetic mean can cancel or
weaken a sparse direction that is highly relevant to the query.

The current `key-direction-adaptive` path replaces that one-vector summary with
a variable number of directional prototypes:

```text
physical retrieval block
    -> contiguous 32-token slices
    -> non-contiguous Key-direction groups inside each slice
    -> 1, 2, or 4 retained prototype means per slice
    -> packed per-layer retrieval index
    -> prototype softmax and logical-block MaxSim
```

The term "semantic segmentation" has a specific operational meaning here. The
method does not parse text, predict message boundaries, or produce contiguous
semantic spans. Instead, it groups tokens whose model-internal content-frame
Keys point in similar directions. These latent groups may be non-contiguous
within the 32-token slice. A precise paper-facing name is therefore
**adaptive Key-direction segmentation** or **latent Key-space semantic
segmentation**, rather than generic text segmentation.

This mechanism is also different from KVMem's separate `round` and `message`
semantic-expansion modes. Adaptive Key-direction v1 does not currently compose
with those modes.

#### 4.9.2 Three levels of granularity

It is important not to conflate three distinct units:

1. **Physical/retrieval block.** This is the unit selected by the host policy
   and transferred by stage-in/stage-out. Its size is `block_tokens`.
2. **Directional slice.** Every physical block is divided into contiguous
   32-token slices. Direction discovery is local to one slice.
3. **Directional prototype.** A slice retains `P in {1, 2, 4}` prototype rows.
   Each prototype row contains one vector for every KV head.

For a full block of `B` tokens:

```text
num_slices = B / 32
prototype_rows_per_layer_per_block =
    sum over slices s of P[l, block, s]
```

Therefore a full 512-token block has 16 directional slices and between 16 and
64 prototype rows per normal-attention layer. Selection and tier movement still
operate on the 512-token physical block; the prototypes only refine its
retrieval score.

The implementation requires:

```text
block_tokens >= 32
block_tokens % 32 == 0
subblock_reduce == max
```

Although the configuration reserves a fixed-four upper bound,
`(block_tokens / 32) * 4`, Adaptive persistent storage is variable length and
contains only the selected rows.

#### 4.9.3 Content-frame Key capture

Direction discovery runs independently for every:

```text
(normal-attention layer, physical block, 32-token slice, KV head)
```

When a prefill chunk produces fresh FP32 `k_batch` rows, the adaptive capture
kernel de-RoPEs each Key at the position at which it was baked:

```text
k_tilde[l,t,g] = deRoPE(K[l,t,g], position_t)
```

Dimensions outside `rope_dim` pass through unchanged. The resulting
`k_tilde` vectors are position-independent content Keys, so clustering does not
change when a selected block is later re-RoPEd to another compact-window
position. Query capture applies the corresponding de-RoPE operation to Q,
placing query vectors and prototypes in the same content frame.

Adaptive capture is incremental across prefill chunks, but the current v1
requires the indexed suffix to begin on a physical-block boundary. A mid-block
adaptive merge is deliberately rejected rather than approximated. The last
block and its last slice may be shorter than the configured full sizes.

#### 4.9.4 Deterministic farthest-first direction discovery

For one layer, slice, and KV head, let:

```text
K = {k_1, ..., k_n},  n <= 32
```

denote the de-RoPEd content Keys. Cosine similarity is used to discover
directions:

```text
cos(a, b) = (a^T b) / (||a||_2 ||b||_2)
```

The first step computes the arithmetic slice mean:

```text
k_mean = (1 / n) * sum_t k_t
```

The first seed is not `k_mean` itself. It is the observed token Key whose
direction is closest to the mean direction:

```text
s_1 = argmax_t cos(k_t, k_mean)
```

This makes the first seed a deterministic medoid-like observed direction while
the `P=1` stored candidate remains the exact full-slice mean.

Each later seed is chosen by farthest-first traversal. Given the current seed
set `{s_1, ..., s_{j-1}}`, the kernel chooses the token whose nearest-seed
cosine is smallest:

```text
s_j =
    argmin_{t not already selected}
    max_{r < j} cos(k_t, k_{s_r})
```

After adding a seed, every token retains the index and cosine of its closest
seed. The procedure adds seed 2, then seeds 3 and 4. It is deterministic for a
fixed input and preferentially exposes Key directions that are poorly covered
by the directions already selected.

This is not iterative k-means:

- there is no random initialization;
- there is no Lloyd iteration;
- assignments are made against the observed seed directions;
- the resulting cluster means are not fed back into another assignment pass.

For method writing, "deterministic farthest-first spherical partitioning
followed by mean aggregation" is more accurate than "adaptive k-means."

#### 4.9.5 Nested `P=1`, `P=2`, and `P=4` candidates

The kernel materializes three candidate representations for every slice:

```text
candidate row 0     : one exact full-slice mean
candidate rows 1-2 : two means under the first two seeds
candidate rows 3-6 : four means under all four seeds
```

This produces seven temporary rows:

```text
1 + 2 + 4 = 7
```

For `P=2` and `P=4`, token assignment is:

```text
a_P(t) = argmax_{j <= P} cos(k_t, k_{s_j})
```

and the stored prototype for direction group `j` is the exact arithmetic mean
of the unnormalized content Keys assigned to it:

```text
C_{P,j} = {t : a_P(t) = j}

c_{P,j} =
    (1 / |C_{P,j}|) * sum_{t in C_{P,j}} k_t
```

Cosine-normalized directions are used only for seeding and assignment. The
stored prototype is not unit-normalized because the retrieval scorer should
retain the magnitude information in the attention-space Key mean. Prototype
storage normally uses FP16; candidate construction, norms, residuals, dot
accumulation, softmax statistics, and final score accumulation use FP32.

The partitions are independent across KV heads. Prototype ordinal `j` for two
different KV heads need not describe the same token subset. However, one
prototype row stores the `j`-th vector for all KV heads, so all heads in one
layer/slice must share the same selected prototype count `P`.

#### 4.9.6 Cosine residual and hierarchical gain rule

For each candidate count `P in {1, 2, 4}`, the capture kernel measures how well
the selected seed directions cover the tokens:

```text
E_P[l,b,s,g] =
    (1 / n) * sum_t
    (1 - max_{j <= P} cos(k_t, k_{s_j}))
```

The implementation clamps numerical overshoot so each contribution is
nonnegative. Empty slices in a partial block are marked with residual `-1` and
are not added to the persistent index.

A subtle but important detail is that `E_P` is a residual to the selected
**seed directions**, not a reconstruction loss to the final prototype means.
The seeds define the directional coverage criterion; the means define the
stored retrieval representation.

For every layer and slice, the host computes two relative residual reductions:

```text
G_1to2 =
    max over KV heads g of
    (E_1[g] - E_2[g]) / max(E_1[g], epsilon)

G_2to4 =
    max over KV heads g of
    (E_2[g] - E_4[g]) / max(E_2[g], epsilon)

epsilon = 1e-8
```

The maximum over KV heads is conservative: if any head has a direction that is
substantially under-represented, the layer/slice retains the larger common
prototype count.

The hierarchical selection rule is:

```text
P = 1

if G_1to2 >= tau_12:
    P = 2
    if G_2to4 >= tau_24:
        P = 4
```

The defaults are:

```text
tau_12 = 0.10
tau_24 = 0.06
```

Consequently:

- the implementation selects only `1`, `2`, or `4`, never `3`;
- `P=4` is considered only after the `1 -> 2` gain passes;
- homogeneous slices remain at one mean;
- slices with one additional useful direction retain two means;
- slices with additional residual structure retain four means.

This is best described as a **hierarchical relative residual-gain rule**. It
models diminishing returns through staged gain tests and a lower second-stage
threshold. It does not explicitly divide gain by the number of added
prototypes: the `2 -> 4` transition adds two rows at once.

#### 4.9.7 Host compaction and persistent packed index

Candidate and residual tensors are bounded scratch for one prefill chunk and
one normal-attention layer:

```text
candidates:
    [chunk_block, slice, 7 candidate rows, KV head, head_dim]

residuals:
    [chunk_block, slice, KV head, {E_1, E_2, E_4}]
```

The scratch is reused by the next layer. After the capture kernel completes,
the executor:

1. synchronizes the completed layer because the scratch will be reused;
2. copies its candidate rows and residuals to host;
3. computes `G_1to2`, `G_2to4`, and the common `P`;
4. copies only the corresponding `1`, `2`, or `4` candidate rows into the
   layer's appendable canonical vector;
5. records each prototype row's parent physical block;
6. records variable `block_offset` and `block_count` metadata;
7. marks the adaptive index dirty and publishes the new suffix according to
   the configured index placement.

The persistent representation is therefore:

```text
layer_values[layer][packed prototype row][KV head][head_dim]
layer_parent[layer][packed prototype row] -> physical block
block_offset[layer, block]                -> first prototype row
block_count[layer, block]                 -> number of prototype rows
```

For a full block with `S = block_tokens / 32` slices:

```text
S <= block_count[layer, block] <= 4S
```

This packed layout is the main efficiency benefit of adaptivity relative to
Fixed-4: the temporary builder always evaluates the bounded seven-row candidate
set, but persistent capacity and online scoring scale with the prototypes that
survive the gain rule.

#### 4.9.8 Query-conditioned prototype scoring

At a reselection boundary, each captured query vector competes against every
included prototype from the corresponding normal-attention layer. Let:

- `l` be a normal-attention layer;
- `m` be a captured query token;
- `h` be a query head;
- `g(h)` be the KV head serving query head `h`;
- `c[l,b,r,g(h)]` be prototype `r` belonging to physical block `b`;
- `A_l` be all prototypes of the retrievable candidate blocks in layer `l`.

The scorer computes attention-temperature logits:

```text
z[l,m,h,b,r] =
    q[l,m,h]^T c[l,b,r,g(h)] / sqrt(head_dim)
```

It then forms one global prototype distribution for every
`(layer, query token, query head)`:

```text
pi[l,m,h,b,r] =
    exp(z[l,m,h,b,r])
    / sum over (b',r') in A_l of exp(z[l,m,h,b',r'])
```

The softmax denominator is over prototypes, not first over blocks and not
independently inside each 32-token slice. When the request is over budget, the
always-kept sink and recent bands are excluded from the competitive denominator
as long as a non-empty retrievable middle remains. The host selector preserves
those bands independently.

For each physical block, the scorer performs a document-side MaxSim over all
of that block's prototypes from all of its 32-token slices:

```text
mass[l,m,h,b] =
    max over r belonging to block b of pi[l,m,h,b,r]
```

The final score is:

```text
score[b] =
    sum over query tokens m
    mean over normal-attention layers l
    mean over query heads h
    mass[l,m,h,b]
```

The implementation sums rather than averages over query tokens. Every block in
one request sees the same query length, so the common factor does not affect its
within-request ranking.

This reduction is deliberately MaxSim rather than prototype-mass summation. A
block wins when at least one of its latent directions matches the query
strongly; unrelated directions in the same block do not average that match
away, and multiple mediocre directions are not simply added to overwhelm one
strong direction in another block. Algebraically:

```text
max_r softmax(z_r)
    = exp(max_r z_r - row_max) / sum_r' exp(z_r' - row_max)
```

so the CUDA scorer can find a block's maximum logit and emit one exponential
per block/distribution after computing the exact global log-sum-exp.

#### 4.9.9 End-to-end control flow

The complete online path is:

```text
configure key-direction-adaptive
    -> require 32-token-aligned block geometry and Max reduction
    -> pre-size per-layer adaptive metadata

for each aligned prefill chunk:
    for each normal-attention layer:
        consume freshly produced FP32 K rows
        -> de-RoPE into the content frame
        -> build 1/2/4 farthest-first candidates per 32-token slice/head
        -> compute E_1/E_2/E_4
        -> copy bounded layer scratch to host
        -> choose P from relative residual gains
        -> append selected means to the packed canonical layer index
        -> publish/upload the dirty layer suffix when GPU placement is active

during the marked query span:
    capture and de-RoPE Q for every normal-attention layer/query head

at semantic reselection:
    finalize only outstanding adaptive index state
    -> mask always-kept bands from competitive scoring
    -> per layer/query token/query head:
         dot against included prototypes
         global softmax over prototypes
         MaxSim-reduce prototypes to physical blocks
    -> sum query-token contributions and average layer/head contributions
    -> copy one score per physical block to host
    -> run the unchanged global physical-block selector
    -> stage in selected physical blocks
    -> assemble and re-RoPE the selected attention window
```

The semantic choice of prototypes is made while the Keys are available during
prefill. The later CPU/GPU index-placement choice changes where packed rows
reside and how they are staged, but not which prototypes exist or how scores
are defined.

#### 4.9.10 GPU and CPU index placement

Adaptive indexing has two production placements:

- **GPU placement:** appendable host vectors remain the canonical construction
  state, while each normal-attention layer has a growable GPU arena. Dirty
  suffixes are uploaded incrementally after capture and can overlap later
  prefill. Query-time finalization normally publishes only small metadata.
- **CPU placement:** pageable host memory remains authoritative. At scoring
  time, bounded layer-sized slots stage the index to GPU. Dot products,
  softmax, and MaxSim still execute on GPU; this is CPU index capacity, not a
  pure CPU retrieval scorer.

The preferred layer-one-pass scorer stores a bounded logits batch, evaluates
each staged prototype dot once, computes the exact layer softmax, and reduces
prototype matches into block scores before reusing the layer slot. The
tiled-two-pass path is retained as a compatibility fallback. GQA-specialized
kernels reuse each KV-head prototype across its associated query heads.

These placement/scorer optimizations preserve retrieval semantics. Their
controlled results, including exact selected-block agreement between CPU and
GPU placement, are recorded in
`docs/kvmem_adaptive_index_optimization_20260730.md`.

#### 4.9.11 Complexity and capacity

Let:

- `L` be the number of normal-attention layers;
- `S` be the total number of non-empty 32-token slices;
- `P[l,s] in {1,2,4}` be the selected count;
- `N = sum_l sum_s P[l,s]` be total persistent prototype rows;
- `H_kv`, `H_q`, and `d` be KV heads, query heads, and head dimension;
- `M` be the captured query length.

Persistent index storage is:

```text
O(N * H_kv * d)
```

where:

```text
L*S <= N <= 4*L*S
```

The single-mean 32-token baseline occupies the lower bound and Fixed-4 occupies
the upper bound. Adaptive storage lies between them according to observed
directional complexity.

Ignoring batching and GQA reuse, prototype scoring work is:

```text
O(M * H_q * d * N)
```

Candidate construction always considers seven temporary rows per
layer/slice/head, but its GPU scratch is bounded by the active prefill chunk and
does not grow with the complete history. Persistent index bytes and retrieval
dot products grow only with selected rows.

#### 4.9.12 Correct interpretation and current boundaries

The implementation supports the following claims:

- it discovers multiple model-internal Key directions inside a local slice;
- it groups potentially non-contiguous tokens without a separate embedding
  model or text parser;
- it adapts representation capacity with a hierarchical residual-gain rule;
- it preserves sparse directional matches through block-side MaxSim;
- it stores and scores only retained prototypes after bounded candidate
  construction.

The implementation does not support stronger claims that:

- it learns explicit natural-language segment boundaries;
- it runs or converges iterative k-means;
- it chooses an arbitrary count from one through four;
- it optimizes an explicit gain-per-added-byte objective;
- it measures residual to the final centroid means;
- each KV head shares the same token membership for prototype ordinal `j`.

Current design trade-offs include:

- the maximum gain across KV heads may retain extra rows because of one head;
- prototype-level global softmax gives a slice with more retained directions
  more competitors in the denominator and more opportunities to produce its
  block's maximum;
- MaxSim retains the strongest localized match but discards the additional
  probability mass of multiple simultaneously relevant directions;
- the adaptive count is query-independent and fixed during prefill;
- v1 requires aligned capture and does not compose with round/message semantic
  expansion.

These are useful ablation dimensions rather than hidden implementation details:
single-mean versus Fixed-4 versus Adaptive, alternative head aggregation,
gain-threshold sweeps, MaxSim versus mass reduction, and query-independent
versus query-aware prototype budgets.

### 4.10 Fallbacks

If query-conditioned mean-k is unavailable, KVMem falls back to other signals:

- A single last-token content scorer if the global content index and query are
  ready.
- Window-local H2O/profile scores if retrieval cannot run.
- Recency-only selection when no learned/model-internal signal is available.

Index construction consumes the freshly produced FP32 `k_batch`, so production
FP8 active KV can use an FP16 Mean-K or Adaptive index; this is the configuration
used by the current long-context evaluations. Legacy paths that attempt to
reconstruct a content index by directly scanning an incompatible quantized
paged-K representation remain unavailable and fall back explicitly.

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
authoritative in DRAM, so demand allocation by itself is not sufficient.

The first raw-K SSD tier was implemented on 2026-07-29 behind
`--kvmem-raw-k-nvme`:

```text
raw-K chunk: CPU-resident | disk-resident | prefetching
V block:     GPU-resident | CPU-resident | disk-resident | prefetching
```

The configured NVMe budget is split at admission: a fixed-capacity anonymous
arena reserves every padded raw-K chunk, and the remainder is used by the
ordinary V tier. Main and local-position MTP raw K share one fixed chunk record.
Completed chunks use a bounded asynchronous write-through pipeline; dirty
partial chunks are persisted before CPU eviction. Full aligned prefill chunks
D2H directly into one of two fixed pinned slots in the final block-major record
layout. Submission records a CUDA transfer fence instead of synchronizing the
host after every chunk. The background writer waits that fence before touching
the pinned bytes. Main and MTP raw-K capture tensors each have two fixed GPU
generations, preventing the next prefill chunk from overwriting a D2H source;
reuse queues its prior fence as an execution-stream dependency instead of
blocking the host. The pageable CPU-cache copy is
published only after the future completes, immediately before that pinned slot
can be reused. At the default 128 MiB slab target, one slot packs three
34 MiB FP8+MTP records.
`write_spans` coalesces adjacent records into one large `pwrite` and one range
writeback while the GPU computes later chunks. A slot is waited only when both
staging slots are owned by SSD, CPU-cache pressure requires recycling one,
or at correctness boundaries such as prefix checkpoint, truncate, reset, and
cold eviction of an in-flight chunk. When the shared CPU budget is full, cold
raw-K chunks can be released instead of making raw-K memory scale with logical
context length.

Selected cold raw K is read by exact physical block range directly into the
existing two-slot pinned/device staging pipeline. A 256-token FP8 block across
16 standard layers is a 4 MiB read; the implementation does not reload its
complete 32 MiB/2048-token raw chunk. The SSD files use the same
open-then-immediate-unlink lifecycle as the V arena and leave no stale cache
after normal exit, exceptions, or `SIGKILL`.

The remaining SSD work is latency/throughput hardening rather than capacity
correctness: replace the bounded `std::async` positional writer with io_uring
fixed-buffer batches for both writeback and scattered raw-block reads, and make
raw-K/V share one global extent planner.

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

Although the paper focus is KV-native retrieval, the implementation also
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

Single-mean retrieval stores block summaries instead of raw token keys:

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

Adaptive Key-direction retrieval preserves this compressed-index principle
while allocating representation capacity according to local directional
complexity. For `S` non-empty 32-token slices and selected counts
`P[l,s] in {1,2,4}`, it stores:

```text
O((sum_l sum_s P[l,s]) * n_kv_heads * head_dim)
```

instead of the full per-token Key tensor. Its bounded builder evaluates seven
candidate rows per active slice, but only selected rows enter the persistent
packed index and later prototype dot products. GPU placement overlaps dirty
suffix uploads with prefill; CPU placement keeps canonical capacity in pageable
host memory and stages complete layers for exact GPU dot/softmax/MaxSim scoring.

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

### 10.9 Three Core Efficiency Optimization Groups

This subsection consolidates the three paper-facing efficiency groups and their
controlled ablation from
`docs/kvmem_performance_evaluation_20260726.md`. It is the implementation-level
definition of the groups. The numerical results remain a historical measurement
of commit `062b403`, not a new measurement of the current worktree.

`--kvmem-optimize-off` controls only these three groups. In particular,
`--kvmem-optimize-off all` does not disable KVMem and does not restore an early
historical implementation. It retains correctness and scalability
infrastructure such as immutable raw-K, the bounded GPU page pool, page-table
assembly, capacity checks, and the common asynchronous transfer machinery.

The four controlled cells are:

| Cell | `--kvmem-optimize-off` | Proactive stage-out | Hierarchical reuse | Packed rematerialization |
|---|---|---:|---:|---:|
| `all-off` | `all` | off | off | off |
| `proactive-only` | `hierarchical-reuse,packed-rematerialization` | on | off | off |
| `proactive-plus-reuse` | `packed-rematerialization` | on | on | off |
| `all-on` | omitted | on | on | on |

Adjacent cells add exactly one group:

```text
all-off
  -> + proactive-stage-out
  -> + hierarchical-reuse
  -> + packed-rematerialization
```

#### 10.9.1 Proactive Stage-Out

**Problem.** Waiting until a pressure boundary to create lower-tier backing
places GPU gather, D2H, CPU admission, and optional SSD persistence directly on
the prefill critical path.

**Implementation.**

- After a completed prefill chunk is safe to read, KVMem starts creating clean
  lower-tier backing for blocks that may subsequently be evicted.
- GPU pages are gathered into a bounded staging buffer and copied in packed D2H
  batches through a fixed number of pinned host slabs.
- CPU-tier admission runs asynchronously. When SSD is configured, persistence
  can continue in background workers without holding the model-compute thread.
- Each block records whether its lower-tier copy is clean and authoritative.
  Pressure eviction can then release an already-backed GPU page by changing
  ownership metadata rather than copying the block again.
- The page cannot be recycled until both its final attention reader and any
  outstanding D2H operation have completed.

This group therefore changes *when* stage-out work is performed and how much of
it is visible at the pressure boundary. It does not reduce the number of blocks
that the next semantic window needs to stage in.

#### 10.9.2 Hierarchical Reuse

**Problem.** Consecutive semantic selections normally overlap. Evicting every
old working-set block and loading every new selected block repeats transfers for
KV pages that are already resident and valid.

**Implementation.**

- `KvMemStore::set_selection` computes the set difference between the previous
  and next selected block IDs.
- Blocks in the intersection retain their physical GPU pages and are not
  included in stage-out, stage-in, or raw-K refresh plans unless their compact
  position requires separate rematerialization.
- Only blocks entering the working set are loaded from a lower tier.
- Clean CPU/SSD backing is inclusive: retaining a GPU page does not invalidate
  its lower-tier copy.
- CPU admission is retrieval/heat/frequency aware. Hot or repeatedly selected
  blocks are preferentially retained in CPU memory, while SSD serves colder
  misses.
- The selected IDs are restored to chronological order before page-table
  assembly, so reuse does not change attention order.

This group primarily reduces transferred *volume*. It is distinct from making
an individual PCIe or SSD transfer faster.

#### 10.9.3 Packed Rematerialization

**Problem.** A selected window may contain thousands of scattered historical
blocks. Per-block CPU copies, H2D calls, scatter operations, and re-RoPE launches
make cold stage-in and assembly launch-bound and synchronization-heavy.

**Implementation.**

- Immutable raw-K uses a block-major host layout so rows for many selected
  blocks can be gathered efficiently.
- Persistent CPU workers gather scattered host blocks into contiguous pinned
  slabs.
- Each slab is transferred with a bulk H2D operation into a bounded reusable
  device buffer.
- CUDA kernels scatter the slab into arbitrary physical KV pages and apply RoPE
  for the new compact positions in batches.
- Double-buffered staging overlaps CPU gather for batch `N+1`, H2D for batch
  `N`, and GPU scatter/re-RoPE for the preceding batch.
- V stage-in can overlap raw-K assembly rather than serializing both streams.
- Adjacent SSD file and buffer spans are coalesced before positional reads.
- A precomputed FP32 RoPE sin/cos table removes repeated `powf`/`sincosf` work
  from the rematerialization kernel.

This group reduces both exposed stage-in time and assembly time. It does not
change the selected block IDs or retrieval scores.

#### 10.9.4 Control and Capability Boundaries

The main control and validation points are:

- CLI parsing: `src/qw3_cli.cpp`;
- default all-on behavior and deprecated legacy-profile mapping:
  `src/qwen_native_backend.cpp`;
- effective-capability checks and `[kvmem-opt-status]` reporting:
  `src/qwen_executor.cpp`;
- selection difference, GPU reuse, and raw-refresh plan construction:
  `src/kvmem_store.cpp`.

In the independent, non-legacy configuration path, a requested group must
either be supported or be reported as not applicable when the complete context
already remains GPU-resident. The implementation must not silently turn an
`all-on` run into a different ablation cell.

The following mechanisms are intentionally outside the three switches because
they are correctness or common scalability foundations shared by every cell:

- bounded GPU KV page allocation and page-table working-set assembly;
- step/chunk-level scheduling and headroom-aware prefill chunking;
- immutable raw-K authority and V-only lower-tier records;
- demand-allocated host storage and bounded reusable transfer slabs;
- split prepare/finish scheduling and the common copy stream;
- incremental Mean-K/Adaptive indexing and scalable page/prototype scorers;
- MTP bounded sibling pages and local-position correctness;
- capacity validation and page-cache bounding.

Likewise, query replay is an accuracy repair that adds compute, and canonical
raw-K rebuilding is a correctness control. Neither is a fourth efficiency
group.

#### 10.9.5 Controlled Measurement at `062b403`

The consolidated measurement used one 515,029-token AgentLongBench sample, a
200K selected-context budget plus a 32K generation reserve, 32-token physical
blocks, FP16 active KV, MTP-4, CPU tiering without SSD reads, and canonical
raw-K reconstruction. Each request contained 12 measured reselection events.
All four cells produced the same correct final answer.

Request-level results:

| Cell | TTFT (s) | Reselection within TTFT (s) | TTFT minus reselection (s) | Post-TTFT generation (s) | Full request (s) | Peak GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| `all-off` | 341.8 | 64.0 | 277.9 | 37.7 | 379.5 | 48,062 |
| `proactive-only` | 337.8 | 58.8 | 279.0 | 37.3 | 375.1 | 48,004 |
| `proactive-plus-reuse` | 309.9 | 30.7 | 279.2 | 37.8 | 347.7 | 48,004 |
| `all-on` | 287.2 | 8.5 | 278.7 | 37.8 | 325.0 | 48,320 |

The non-reselection portion of TTFT stayed within 277.9--279.2 seconds, which
indicates that the model-prefill workload was stable across the four cells.
From `all-off` to `all-on`:

- reselection fell from 63.957 s to 8.488 s, an 86.73% reduction or 7.53x
  working-set-management speedup;
- TTFT fell from 341.844 s to 287.176 s, a 15.99% reduction;
- full-request latency fell from 379.530 s to 324.973 s, a 14.37% reduction;
- peak GPU memory increased by 258 MiB, or 0.54%.

Reselection breakdown:

| Cell | Selection (s) | Stage-out (s) | Stage-in wall (s) | Assembly (s) | Reselection total (s) | GPU reuse / natural overlap |
|---|---:|---:|---:|---:|---:|---:|
| `all-off` | 0.033 | 3.645 | 29.039 | 31.239 | 63.957 | 0% |
| `proactive-only` | 0.039 | 0.268 | 28.995 | 28.992 | 58.843 | 0% |
| `proactive-plus-reuse` | 0.053 | 0.042 | 1.613 | 28.478 | 30.730 | 100% |
| `all-on` | 0.051 | 0.050 | 0.310 | 7.901 | 8.488 | 100% |

The adjacent-cell attribution is:

- **Proactive Stage-out:** stage-out fell from 3.645 s to 0.268 s, a 92.65%
  reduction or 13.60x speedup. Stage-in remained about 29 seconds because this
  group does not change the next working set.
- **Hierarchical Reuse:** stage-in fell from 28.995 s to 1.613 s, a 94.44%
  reduction. All 58,417 naturally overlapping block events remained on GPU,
  and stage-in block events fell from 76,798 to 18,381.
- **Packed Rematerialization:** assembly fell from 28.478 s to 7.901 s, a
  72.26% reduction or 3.60x speedup; exposed stage-in wall time also fell from
  1.613 s to 0.310 s.

Even after all three groups were enabled, assembly accounted for approximately
93.1% of measured reselection time in this canonical raw-K experiment. It was
therefore the dominant remaining bottleneck for that historical configuration.
This observation must not be combined directly with later bounded re-RoPE,
FP8, different block sizes, or current runtime-workspace optimizations without
a new controlled run.

Detailed commands, hardware, raw result paths, metric definitions, and
limitations remain in
`docs/kvmem_performance_evaluation_20260726.md`. Supporting focused experiments
are:

- `docs/kvmem_cpu_proactive_writeback_benchmark_20260725.md`;
- `docs/kvmem_cpu_transfer_optimization_benchmark_20260724.md`;
- `docs/kvmem_assembly_rerope_optimization_benchmark_20260724.md`;
- `docs/kvmem_ssd_writeback_benchmark_20260724.md`.

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

One mean can under-rank a slice where the relevant evidence is sparse. Smaller
blocks or larger budgets reduce the problem, and Adaptive Key-direction
segmentation mitigates it by retaining one, two, or four local direction means
and applying block-side MaxSim. It does not eliminate approximation: grouping
still occurs inside fixed 32-token slices, prototype counts are capped at four,
the count is selected before the query is known, and MaxSim preserves only the
strongest prototype mass for each query distribution. Exact-mass retrieval
avoids prototype compression more directly but uses much more memory.

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

Construct a content-frame Key index and score all historical blocks by
query-conditioned softmax mass. The compact baseline stores contiguous mean
Keys. Adaptive Key-direction segmentation divides each physical block into
32-token slices, discovers non-contiguous directional groups with deterministic
farthest-first traversal, and retains one, two, or four prototype means according
to hierarchical cosine-residual gains. Use all standard-attention layers and
all captured query tokens; reduce a block's variable prototype run with MaxSim.
Optionally support raw-key exact-mass retrieval for higher fidelity.

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

Adaptive prototype construction:

```text
s_1 = argmax_t cos(k_t, mean_t k_t)

s_j = argmin_t max_{r<j} cos(k_t, k_{s_r})

E_P = mean_t (1 - max_{j<=P} cos(k_t, k_{s_j}))

P in {1,2,4}, selected by relative E_1->E_2 and E_2->E_4 gains
```

Adaptive block score:

```text
pi[l,m,h,b,r] =
    softmax over all included layer-l prototypes
    (q[l,m,h]^T c[l,b,r,g(h)] / sqrt(d))

r_b = sum_m mean_l mean_h max_{r in block b} pi[l,m,h,b,r]
```

Key implementation points:

- `kvmem_capture_kbar_multi`
- `kvmem_capture_adaptive_layer`
- `kvmem_capture_query_multi`
- `block_kmean_content_batch_kernel`
- `block_kdirection_adaptive_batch_kernel`
- `block_attn_score_softmax_pages_kernel`
- `block_attn_adaptive_accumulate_logits_kernel`
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
selects a bounded set of historical KV blocks using query-conditioned
content-Key retrieval, optionally representing each 32-token slice with an
adaptive set of latent Key-direction prototypes, assembles the selected blocks
as a compact attention window through page-table aliasing and in-place re-RoPE,
and preserves the remaining KV state across a GPU/CPU/NVMe hierarchy with
canonical lower-tier storage.
