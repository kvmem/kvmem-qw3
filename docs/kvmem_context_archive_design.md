# KVMem Context Archive (KCA) design

Status: implementation in progress. See
[`kvmem_context_archive_status.md`](kvmem_context_archive_status.md) for what
is coded, what was measured on a 32K archive, and what is still open.
Supersedes nothing; the three existing in-process caches
(`QW3_PREFIX_CACHE`, KVMem warm P/M checkpoints, named local cache) remain and
are unaffected when the archive is not requested.

Scope for the first implementation:

- FP8 (`QW3_KV_DTYPE=fp8`, e4m3, no row scale) KV only. FP16/FP32/q8 archives
  are rejected at open, not silently reinterpreted.
- `--kvmem-immutable-k` required. This is not an extra constraint: KVMem already
  refuses fp8 without it (`qwen_executor.cpp:7436`).
- Qwen3.6-27B `qwen35` hybrid layout: 64 main layers, `full_attention_interval=4`
  → 16 standard-attention layers, 48 DeltaNet layers.

## 1. Why the current caches cannot do this

All three existing mechanisms store a `QwenExecutor::StateSnapshot`, which is a
set of device-to-device clones of the recurrent/conv/hidden tensors. None of
them own the KV bytes; those stay in the live GPU page pool, pinned CPU tier,
and NVMe tier. Three consequences follow:

1. A checkpoint is a bookmark into mutable shared state. Any unrelated request
   that calls `reset_state()` invalidates it, which is why
   `invalidate_local_kvmem_caches("evicted")` must be called on every non-cache
   request path. Only one executor lineage can be ready at a time.
2. The NVMe backing file is `unlink`ed immediately after `open`
   (`nvme_kv_tier.hpp:97`) and is documented as an ephemeral cache. Nothing
   survives a restart by construction.
3. `local_cache_fingerprint()` mixes `kvmem_budget`, `kvmem_gen_budget`,
   `kvmem_retrieval_method`, and `kvmem_index_placement` into the identity of a
   checkpoint. Those are policy, not layout: changing any of them makes an
   otherwise valid checkpoint unusable.

There is also a structural gap independent of implementation quality. DeltaNet
recurrent state is the result of a sequential scan whose per-token update
`S(I − βkkᵀ) + B` is neither commutative nor invertible. KV can be truncated to
a prefix by dropping blocks; recurrent state cannot. A single end-of-context
snapshot therefore cannot serve a prefix-truncation experiment at all.

## 2. Persist versus derive

The archive stores only bytes that cannot be reproduced without re-running the
model. Everything else is derived at attach time.

| Data | Decision | Rationale |
|---|---|---|
| Raw (content-frame, position-free) K | persist | Irreducible. Writer already exists (`kvmem_write_raw_k`) |
| V | persist | Irreducible. V carries no RoPE, so bytes stay valid forever |
| Recurrent + conv + hidden at ladder positions | persist | Not truncatable, not invertible |
| Token IDs | persist | Integrity check, re-render, re-prefill fallback |
| Mean-K / sub-block / adaptive prototypes | derive | Policy-dependent: `prototype_mode`, `n_subblocks`, and the adaptive gain thresholds all change it. Persisting it would pin those into the archive |
| `KvMemBlock` table | derive | Fully determined by `total_tokens` and `block_tokens` |
| `baked_pos`, `remap_count`, `remap_abs_delta` | derive | Attach is cold; every block takes the `raw_refresh` path |
| GPU page assignment, window page table, working set | derive | Rebuilt by the first reselect |
| `attn_score` / `profile_score` / `retrieval_score` | derive | Re-scored per query |
| Tier slot mapping | derive | Direct-mapped: chunk/block id is the slot |

## 3. On-disk layout

```
<archive>/
  manifest.json      layout_key, policy_snapshot, ladder index, seal state
  valid.chunks       per raw-K-chunk validity, for resumable build
  valid.blocks       per V-block validity, for resumable build
  rawk.bin           offset = chunk_id * raw_chunk_bytes
  v.bin              offset = block_id  * v_block_bytes
  tokens.bin         u32 token ids
  state/<pos>.bin    recurrent + conv + h at ladder position <pos>
```

There is deliberately no retrieval index in the archive. See §5.

`rawk.bin` uses the existing block-major chunk layout as its physical format.
Each record stores the main standard-attention raw K followed by the optional
MTP raw-K segment. With
`kvmem_raw_k_chunk_tokens_ = 2048`, `raw_k_row_bytes = n_kv_heads * head_dim *
elem_size = 4 * 256 * 1 = 1024`, and 16 raw layers, one chunk is 32 MiB.

`v.bin` replaces the LRU slot table with a direct map so no slot index has to be
persisted. One V block at `block_tokens=128` is `16 * 128 * 4 * 256 * 1` = 2 MiB.

The MTP layer's raw K and V are archived only when the archive was built with
MTP enabled; this is recorded in `layout_key` and costs one extra layer
(+1/16 on both `rawk.bin` and `v.bin`).

## 4. Attach, not deserialize

Opening an archive must not read the payload. The attach path is:

1. Validate `layout_key` against the running engine; refuse on any mismatch.
2. Rebuild the block table from `total_tokens` and `block_tokens` (host-only,
   microseconds).
3. Mark every block `tier=SSD`, `nvme_slot=block_id`, `ssd_clean=true`,
   `baked_pos=-1`.
4. Read one `state/<pos>.bin` and load it as the executor's recurrent/conv/h.
5. Build the retrieval index by streaming `rawk.bin` (§5).

The first reselect then streams only the selected window through the existing
stage-in path. This is the same cost the steady-state path already pays; attach
adds no new bulk transfer beyond the index pass. Because every block is cold,
all K is rebuilt via `raw_refresh`, which means an attached session starts with
a *cleaner* numerical state than a long-lived live session that has accumulated
re-RoPE rotations.

## 5. Retrieval index is rebuilt, never archived

The mean-K / sub-block / adaptive prototype index is a function of raw K *and*
of the retrieval policy. `prototype_mode`, `n_subblocks`,
`adaptive_gain_1to2`, and `adaptive_gain_2to4` all change its contents and its
packing. Archiving it would force those parameters into `layout_key` and make
index ablations on a shared archive impossible, which is the exact failure mode
§8 exists to prevent. It is therefore always rebuilt.

The rebuild is a dedicated bulk pass rather than the incremental
capture the live prefill path uses, and it is cheaper per byte than that path:
`rawk.bin` is already in the content frame, so no de-RoPE is required before the
mean/prototype kernel. The pass streams chunks from SSD, copies them H2D, and
reduces them.

The first implementation read the block-major record one layer at a time. In
the reference layout that turned every 32-MiB chunk into 256 strided 128-KiB
`pread` calls. The current implementation instead reads each complete main-K
chunk once, gathers one layer at a time in pinned host memory, and leaves the
H2D/widen/prototype kernels unchanged. A 12K selected-set parity test passes;
the 1.704M-prefix resume measured 34.4 seconds (32.1 seconds in reads), or
about 20.2 seconds per 1M tokens versus the old 36.7-second baseline.

The initial storage-bandwidth estimate was too optimistic. The original 1M
layer-strided implementation took **36.7 s**. Contiguous chunk reads reduce the
same archive to **22.9 s** (21.5 s read, 0.7 s host gather, 0.8 s other); the
preallocation fix should further help newly-built, less-fragmented archives.
A 2M fragmented archive takes 42.7 s, so current 10M extrapolation is roughly
3.5 minutes rather than 46 seconds. A process still pays this once per attach,
not once per question. Double-buffered/parallel reads, overlap with GPU work, or
an opt-in policy-keyed derived sidecar outside the sealed archive remain real
phase-4 optimizations.

Two consequences worth planning around. Index *runtime memory* is unchanged by
this decision and is the real constraint: at `block_tokens=128` a single
prototype per block is 1.2 GiB at 10M, `block_tokens=32` is 4.8 GiB, and
`KeyDirectionFixed4`/`KeyDirectionAdaptive` reach roughly 19 GiB, which is what
`KvMemIndexPlacement::CPU` exists for. And if the rebuild becomes a bottleneck
across hundreds of short attaches, the answer is an opt-in derived-cache sidecar
*outside* the archive directory, keyed by a hash of the index parameters, with
the archive remaining the sole authority. That is a pure performance option and
is not part of the archive contract.

## 6. Ladder and prefix truncation

Ladder positions must be aligned to `lcm(block_tokens, 2048)`; the raw-K chunk
size of 2048 tokens dominates for every supported `block_tokens`. The default
interval is 131072 tokens (64 chunks), which satisfies the alignment rule.

Truncating an archive to `N` tokens:

1. Pick the largest ladder position `p <= N`.
2. Load `state/<p>.bin`.
3. Set the block table to `N / block_tokens` blocks.
4. Re-prefill the residual `N - p` tokens from `tokens.bin`.

Residual work is bounded by the ladder interval. Truncation itself allocates
nothing and copies nothing: the base archive is untouched and shared.

Only block-aligned prefixes are sealed, so the mean-K index has no partial-block
entries. This sidesteps KVMI-002 for archived contexts.

## 7. Copy-on-write branching

The base archive is opened `O_RDONLY`. A request owns an unlink-after-open
sparse overlay outside the archive directory, with the same direct-mapped slot
offsets as the base. Reads use the base until a slot diverges; writes use
copy-on-write. Because a physical record contains main and MTP segments, the
first partial-slot write copies the complete base slot before patching it and
atomically publishing the overlay slot. This is required not only for appended
`block_id >= base_block_count`, but also when residual/query execution rewrites
a base slot during window reconstruction. Resetting to base discards the
ephemeral overlay and reloads the ladder state; nothing in the archive is
mutated.

This removes the "one ready lineage at a time" restriction. Several branches can
exist in one process, and several processes with different optimization flags
can attach the same archive concurrently.

## 8. layout_key versus policy_snapshot

`layout_key` must match exactly, because it determines byte layout or numerics:

```
complete model GGUF SHA-256, architecture, n_layers, full_attention_interval,
standard-attention layer set, n_kv_heads, head_dim, head_v_dim,
kv dtype (fp8_e4m3), rope_dim, rope_theta, block_tokens,
raw_chunk_tokens, immutable_source_k, mtp_archived
```

`policy_snapshot` is recorded for provenance but never checked:

```
select_budget, gen_budget, sink_blocks, recent_blocks, select_method,
retrieval_method, prototype_mode, n_subblocks, adaptive_gain_1to2,
adaptive_gain_2to4, subblock_reduce, update_mode, interval,
index_placement, adaptive_score_mode, gpu_memory_ratio,
cpu_tier_bytes, nvme_tier_bytes, numa_policy, optimize_* flags,
mtp_enabled at run time
```

The complete GGUF digest is cached outside the archive using canonical path,
device/inode, size, mtime and ctime as the cache validator. New format-v3
archives require its 64 hexadecimal characters; v1/v2 archives remain readable
only through an explicit legacy path because their manifests recorded file size
but no content digest. Builds with OpenSSL use its optimized SHA-256 path and
retain the portable implementation as a fallback. On the current host a cold
25+ GiB model scan added about 23 seconds; a validated cache hit avoided the
scan entirely.

Every retrieval-index parameter is on the policy side precisely because the
index is rebuilt rather than archived (§5).

Everything in `policy_snapshot` is free to differ between build and attach, and
between two concurrent attaches. This is what makes same-context optimization
A/B possible.

## 9. Build: streaming and resumable

The archive writer is the existing writeback path retargeted at the archive
files, not a separate serialization pass; a 10M context cannot be buffered.

- Raw-K chunks are written by `kvmem_persist_completed_raw_k_chunks` as they
  complete.
- V is written when a block stages out. Blocks still GPU-resident at seal time
  are flushed explicitly.
- A new build uses `posix_fallocate` for both direct-mapped arenas before the
  first write. This prevents interleaved raw-K/V growth from fragmenting the
  two large files and fails early on insufficient capacity. Filesystems that
  do not support preallocation warn and fall back to sparse growth.
- At each ladder position, `capture_state` is followed by a D2H dump to
  `state/<pos>.bin`.
- Before publishing a ladder, the writer materializes the deterministic
  sink+recent selection from immutable raw-K/V. A resumed writer reconstructs
  the same canonical attention frame before ingesting the suffix; otherwise a
  live dense page layout and a restored compact page layout can diverge
  numerically even when their stored KV bytes are identical.
- `valid.chunks` / `valid.blocks` are updated after completed writes;
  `manifest.json` is written
  last and is the commit point.

This ordering has been validated for process death/SIGKILL and restart on the
same host. It is not yet a power-loss contract: claiming recovery across host
power failure additionally requires fsync of raw-K/V, tokens, state, bitmaps,
the renamed manifest, and the archive directory in that order.

On resume, the input token prefix is compared with `tokens.bin`. Bytes appended
after the last durable ladder are truncated before the suffix is written again.
This prevents a crash between token append and manifest commit from shifting
the token stream.

If a multi-hour ingest dies at 6M tokens, the bitmap and the last sealed ladder
position allow resuming instead of restarting.

## 10. Measured cost model

Qwen3.6-27B, fp8 e4m3, 16 standard-attention layers plus one archived MTP
attention layer, `n_kv_heads=4`, `head_dim=256`, `block_tokens=128`. Recurrent snapshot is f32 regardless of KV
dtype: 48 layers x 48 v-heads x 128 x 128 x 4 B = 144 MiB, plus conv, measured
at 157,811,352 B (150.5 MiB) in KVMI-012.

| Component | Per unit | At 10M tokens |
|---|---|---|
| Main raw K + MTP raw K | 17 KiB/token | 162.1 GiB |
| Main V + MTP V | 17 KiB/token | 162.1 GiB |
| Ladder @ 131072 | 150.5 MiB/point | 11.2 GiB (76 points) |
| Tokens | 4 B/token | 40 MB |
| **Total** | **34 KiB/token + ladder** | **≈335 GiB** |

Scaling including the proportional ladder cost: 1M ≈ 33.5 GiB, 2M ≈ 67 GiB,
4M ≈ 134 GiB. The host SSD had ~669 GiB free as of 2026-08-01, so a 10M archive occupies about 50% and leaves room for a
second smaller archive.

Archive size is independent of `block_tokens` and of every retrieval-index
setting, because the index is not stored. `block_tokens` only changes the
granularity of `v.bin` records and the runtime index memory discussed in §5.

## 11. Validation

- Byte parity: for a 2M archive, the selected block set and the assembled window
  contents after attach must equal a cold live run with identical policy.
- Output parity: greedy decode from an attached archive must match the live run
  token for token.
- Truncation parity: attach-and-truncate to `N` must match a fresh build whose
  input is the first `N` tokens.
- Branch isolation: `k` sequential questions against one archive must each
  produce the same answer as `k` independent attaches.
- Crash/resume parity: with identical ladder cadence, token stream, main/MTP
  raw-K and V, final recurrent state, selected blocks and greedy output must
  match an uninterrupted build.
- FP8 scorer check (blocking, run before any large build): confirm
  `[kvmem-scorer] requested=mean-k used=mean-k fallback=0` under fp8. The typed
  de-RoPE/mean kernel is a full retrieval path at fp8
  (`qwen_executor.cpp:13573`), which makes the fp8 caveat in KVMI-009 stale, but
  this must be confirmed end to end on the archive path.

## 12. Phasing

1. Archive writer, reader, ladder/truncation, copy-on-write branching and bulk
   index rebuild. **Complete.**
2. Resumable build and full main/MTP byte parity. **Complete.**
3. Dedicated frozen Serve API, v3 model digest and 1M validation. **Complete.**
4. 2M SIGKILL/resume/attach validation and first index-I/O optimization.
   **Complete.**
5. Large preallocated-arena A/B and the 10M scale experiment.

Splitting the older process-local `local_cache_fingerprint` into layout and
policy identities remains useful for that separate cache, but no longer gates
archive A/B: archive layout and policy are already separate in the manifest.

## 13. Open risks

- KVMI-012 (selected attention window and recurrent state describe different
  histories) is not fixed by this design. The archive does make it much cheaper
  to study: swapping a different history's recurrent state becomes a 150 MiB
  read instead of a multi-hour re-prefill.
- Direct-mapped `v.bin` assumes the archive can hold every block. A partially
  built archive is expressed through `valid.chunks` / `valid.blocks`, not through slot reuse; the
  tier's LRU path is bypassed entirely for archived blocks.
- Concurrent read-only attaches share the kernel page cache. With
  `drop_page_cache` enabled the tier already advises `DONTNEED` after reads, so
  two attached processes should be measured for read amplification before
  treating concurrent A/B as free.
- The current fragmented-archive index rebuild extrapolates to roughly 3.5 minutes at 10M and competes with the
  first reselect's stage-in for SSD bandwidth. Whether the two passes should be
  fused (score blocks as their raw K streams past) is an open optimization, not
  a correctness question.
- Crash/resume parity, including the appended MTP raw-K/V record segments, is
  byte-identical after restoring `MtpPrefixHidden` into an allocated runtime
  tensor before the resumed suffix starts.
