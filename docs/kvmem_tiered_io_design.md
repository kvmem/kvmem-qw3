# KVMem tiered KV-cache I/O design

How qw3 keeps a multi-million-token KV cache addressable while only a bounded
working set is GPU-resident, and how the GPU↔CPU↔NVMe transfers are hidden behind
compute. All file:line references are into `src/qwen_executor.cpp` unless noted.

## 1. Problem

A long agentic session prefills far more KV than fits in GPU memory. At
block-tokens = 256 and fp16, one block is ~16 MiB (KV ≈ 64 KiB/token: 16
standard-attention layers × n_kv_heads × head_dim × 2 for K+V × 2 bytes). A 2M-token
context is ~128 GiB of KV — it cannot live on a 96 GiB card, and even if it could,
attending over all of it would make latency grow linearly with context.

KVMem's answer: **retain the full KV across a GPU→CPU→NVMe tier hierarchy, but only
assemble a bounded window (`--kvmem-budget`, e.g. 32768 tokens = 128 blocks + 1 sink)
onto the GPU per step.** The tiering machinery moves blocks between tiers cheaply
enough that per-step latency stays flat as the total context grows. Empirically the
GPU-resident set is constant at 129 blocks / ~2.02 GiB from 128K to 2M, and prefill
throughput stays ~3200 tok/s (see §9).

## 2. Tiers

| Tier | Backing | Role | Allocated when |
| --- | --- | --- | --- |
| **GPU page pool** | Device paged KV (`kv_pages_`), capped at `--kvmem-gpu-memory-ratio` of the card | Holds the assembled window + blocks not yet evicted | always |
| **CPU tier** | Pinned host (`cudaHostAlloc`), `--kvmem-cpu-gb` | First spill target; fast async H2D/D2H | max ladder target ≥ tier-threshold |
| **NVMe tier** | Files under `--kvmem-nvme-dir`, `--kvmem-nvme-gb` | Overflow when the CPU tier is full | max ladder target ≥ tier-threshold |

Each committed block carries a tier tag (`KvTier::{GPU,CPU,SSD}`) plus a CPU slot
and/or NVMe slot in the block store. `kvmem_tier_usage()` reports live per-tier bytes;
`QW3_KVMEM_TIMING=1` wall-clock-attributes every micro-step and
`QW3_KVMEM_TRACE=1` logs each block move (`[kvmem-tier] …`).

## 3. Bounded GPU page pool and in-prefill offload

The GPU page pool is a fixed set of physical KV pages. As a prefill chunk streams KV
in, `kvmem_register_append` / `kvmem_register_until` (:2842 / :2850) register the newly
written tokens as store blocks. Before each chunk claims its pages,
`kvmem_maybe_prefill_offload` (:2857) checks the pool's free-page count against the
headroom the *next* chunk needs (`next_chunk_pages + keep_free_blocks × pages_per_block`,
:2873-2876). When free pages fall below that, it triggers a reselect (:2878), which
evicts the blocks outside the working set to CPU/NVMe **while prefill is still running**.

This is the key to flat throughput: instead of a stop-the-world eviction at the
prefill→decode boundary, eviction is spread incrementally across prefill compute, and
the resident set never exceeds the pool. The cost is real (it is disk/PCIe traffic) and
is **folded into `prefill_s`** — see §8 — which is why the 2M turn's prefill dips
(88 s of its 383 s wall is in-prefill offload).

## 4. Stage-out (eviction): GPU → CPU → NVMe

`kvmem_stage_out(block_ids)` (:3263) evicts each block:

1. **D2H into pinned host** (:3275-3282). The block's K/V pages are copied device→host
   on the dedicated KV copy stream (`begin_kv_transfer_from_device` … `wait_kv_transfer`).
   The destination is a pinned staging buffer (`kvmem_ensure_stage_pinned`) so the copy
   goes async rather than serializing through the driver's pageable bounce buffer.
2. **Place in the CPU tier** (:3286-3324). `place_block_evicting` finds a CPU slot; if the
   CPU tier is full it returns a **victim** block, which is written down to NVMe
   (`write_block`, :3304) before the incoming block takes its slot (`memcpy`, :3321).
   This is the CPU→NVMe cascade — NVMe only fills after the CPU tier is saturated.
3. **CPU-less config** (:3325-3328): if there is no CPU tier, the block is written
   straight to NVMe.
4. **Reclaim GPU pages** (:3336). The block's logical pages are released back to the pool
   (`release_logical_pages`), and the block store is updated with the new tier + slot.

## 5. Stage-in (prefetch): CPU/NVMe → GPU, split for overlap

Stage-in is deliberately split into two halves so I/O can overlap with compute.
`kvmem_stage_in` (:3101) is just `start_prefetch` immediately followed by
`finish_prefetch`; the overlap-aware callers keep them apart (see §6).

**`kvmem_start_prefetch(plan)` (:3106) — issue, don't wait:**
- Opens the KV copy stream (`begin_kv_transfer_to_device`, :3117).
- For each block in the window that isn't already resident:
  - **CPU-tier block**: queue an async H2D copy from its pinned slot on the copy stream
    (`kvmem_copy_block_from_host` → `copy_bytes_from_host_async`), :3170.
  - **NVMe block**: record a read request (block id, byte count, host buffer), :3147-3151.
- Kick off all NVMe reads on a **background thread** via
  `std::async(std::launch::async, …)` (:3186) — the disk reads run off the critical path.
- Return without synchronizing. `kvmem_prefetch_.active` stays true.

**`kvmem_finish_prefetch()` (:3204) — join and land:**
- `nvme_future.get()` joins the background disk reads (:3211).
- Issue the H2D copies for the NVMe-read buffers on the copy stream (:3213-3235).
- `wait_kv_transfer()` (:3238) synchronizes the copy stream so all K/V is on-device.
- Release the vacated CPU/NVMe slots and mark blocks `KvTier::GPU` (:3241-3248).

## 6. Overlapping I/O with the MTP prefix rebuild

The reselect at the prefill→decode boundary is itself split:

- `kvmem_prepare_reselect()` (:3541): score blocks with the CLI-selected scorer
  (`--kvmem-retrieval-method`: `mean-k` softmax-over-pages default, or `per-token`
  ExactMass; :3586-3593), pick top-k (:3604), **evict first** (:3624), then
  `start_prefetch` (:3625) and return with `kvmem_pending_reselect_ = true`.
- `kvmem_finish_reselect()` (:3630): `finish_prefetch` (:3632) then `kvmem_assemble`
  (:3633, builds the window page table + re-RoPE + k-bar).

Between those two calls, the MTP speculative decoder rebuilds its accepted-token prefix
on the GPU (`rebuild_accepted_mtp_prefix` / `rebuild_current_mtp_prefix`,
`qwen_native_backend.cpp:7561` / `:7583`, which call `kvmem_finish_reselect` with
`finish_deferred_kvmem=true`, :7566 / :7581 / :7587 / :7601). So the **NVMe disk reads
(background thread) and the CPU→GPU H2D copies (copy stream) run concurrently with the
MTP prefix-rebuild compute** — the stage-in latency is hidden behind work that had to
happen anyway. `kvmem_reselect()` (:3536) is the non-overlapped convenience wrapper
(prepare + finish back-to-back) used where there is no compute to hide behind.

## 7. Two ordering / stream invariants

- **Evict before stage-in** (:3613-3624). A query-conditioned reselect can resurrect a
  large, scattered set of blocks in one shot. If `start_prefetch` allocated pages for the
  stage-ins before `stage_out` freed the evicted ones, the bounded pool (left near-full by
  prefill) could overflow. Running `stage_out` first guarantees the post-eviction resident
  set is ≤ the window (≤ budget ≤ pool), so the stage-in always fits. The two sets are
  disjoint (stage_out = resident − window, stage_in = window).
- **Dedicated copy stream with event ordering** (`kernels_cuda.cu:4165-4184`). Transfers
  run on `kv_copy_stream_`, separate from `exec_stream_`, so they overlap compute.
  `begin_kv_transfer_from_device` records an event on the exec stream and makes the copy
  stream wait on it (:4168-4172) — the D2H cannot start until the KV it reads has actually
  been produced. `wait_kv_transfer` is a `cudaStreamSynchronize` on the copy stream only.

## 8. Timing attribution (what counts against prefill throughput)

Two distinct classes of tier I/O, charged differently by `run_kvmem_session`
(`qwen_native_backend.cpp:1311-1343`):

- **Boundary reselect** steps 1–4 (selection / stage-in / stage-out / assemble): the
  `sel_ms` / `in_ms` / `out_ms` / `asm_ms` columns. Charged **separately**, diffed against
  the prefill→decode boundary snapshot — **not** in `pre_tok/s`.
- **In-prefill forced offload** (§3): the eviction fired *during* prefill by
  `kvmem_maybe_prefill_offload`. Folded **into `prefill_s`**, so it **does** count against
  `pre_tok/s`. Surfaced as `inprefill_offload_ms` so the prefill wall isn't opaque.

So: the boundary stage numbers are excluded from prefill throughput; the in-prefill
offload is included (and dominates the 2M prefill dip).

## 9. Empirical behavior (128K → 2M growth benchmark)

`scripts/kvmem_session_profile.py`, block-tokens 256, window 32768, gpu-ratio 0.5,
cpu 64 / nvme 512 GiB, MTP chain 4, one persistent process growing the context:

| ctx | total KV | GPU-resident | CPU | NVMe | prefill tok/s | in-prefill offload |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K | 8 GiB | 2.02 GiB | 6 | 0 | 3382 | 3.0 s |
| 256K | 16 GiB | 2.02 GiB | 14 | 0 | 3212 | 3.9 s |
| 512K | 32 GiB | 2.02 GiB | 30 | 0 | 3196 | 8.3 s |
| 1M | 64 GiB | 2.02 GiB | 62 | 0 | 3208 | 16.0 s |
| 2M | 128 GiB | 2.02 GiB | 64 | 62 | 2738 | 88.1 s |

- GPU-resident KV is **constant at 129 blocks / 2.02 GiB** across a 16× context growth.
- Boundary stage-in/out are sub-millisecond here (the window stays resident under the
  recency-dominated synthetic corpus); the visible I/O is the in-prefill offload.
- NVMe engages only at 2M, after the 64 GiB CPU tier fills (62 GiB spills to disk).
- Peak process GPU 46.3 GiB (≤48 target), host RSS ~95 GiB.
- Prefill stays ~3200 tok/s until 2M, where 88 s of the 383 s prefill wall is offload —
  the tiering cost becomes visible only once NVMe is in the loop.

## 10. Reproduce

```
python3 scripts/kvmem_session_profile.py \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ladder 128K,256K,512K,1M,2M \
  --block-tokens 256 --window 32768 \
  --gpu-ratio 0.5 --cpu-gb 64 --nvme-gb 512 \
  --nvme-dir /data/qw3_kvmem_eval_nvme \
  --chain 4 --decode-tokens 256 \
  --out-json /data/chaidi/kvmem_eval/results/kvmem_growth_128k_2m.json
```

Add `--temp 0.6` to profile the sampled decode path (prefill/tiering unchanged; only
decode slows, from the host-side sampling round-trip — see the decode-path note below).

## 11. Related: decode-path host round-trip under sampling

Orthogonal to tiering, but relevant to the same benchmark. Greedy decode picks the token
on-device (`backend_.argmax`, `qwen_executor.cpp:2096` / `:2060`) and returns only a
scalar. Sampling (`temp>0`) needs the full distribution, so the per-row logits are copied
device→host (`row_logits_host`, :2073-2079) and temperature / penalties / top-k / top-p /
multinomial run on the CPU (`sample_token`). For MTP this is per verify row (the
speculative-sampling accept test needs each row's full target distribution). At ~152K
vocab that is ~0.6 MB/row over PCIe plus a host sort — the few-ms/token that makes the
sampled decode 15–44% slower than greedy. It is an implementation choice (host code reuse
across plain and MTP decode), not a hard requirement; on-device sampling would remove the
round-trip.
