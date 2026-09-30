# Shared KVMem page quotas and MTP window execution — 2026-09-25

The persistent DFlash goal remains **active and incomplete**. This stage removes
the shared-arena allocator obstacle for Q8_0 target weights and E4M3 FP8 KV.
It does not yet admit DFlash into the continuous-batching scheduler. No BF16
target loading or BF16 KV support is added.

## Allocation and lifetime

`SharedKvPagePool` reserves each request's resident quota from the global arena
at admission. Freed resident pages remain within that lease until teardown;
another request cannot consume the working-set guarantee. Reversible private
queries may temporarily borrow additional pages, bounded by their reserve and
the parent's availability. Partial admission failure returns the acquired lease.
Cancellation and teardown return pages through the global allocator, which
continues to honor prefix-cache pins.

The allocator interface now separates quota (`total_pages`) from the valid
physical ID range (`physical_page_capacity`). High global IDs remain valid in a
small per-request lease. Main and MTP configure/teardown preserve their external
parent allocators and use separate leases over the existing global tensors.
Immutable raw-K authority, bounded tiering and MTP local positions remain enabled.
Ordinary CB prefill delegates allocation to the KVMem controller's internal
chunks; allocating the whole scheduler chunk first exhausted its resident quota.

The host page-table test exercises high IDs, separate requests, resident
exhaustion, 100 private borrow/rollback cycles, foreign releases, partial
admission failure, teardown and global prefix pins. This is allocator ownership
coverage, not an end-to-end KVMem prefix-reuse qualification.

## MTP execution corrections

Joint draft attention now uses the request's selected MTP window page list,
window-local write/RoPE positions and true logical allocation position. Temporary
window extensions are restored on return; speculative drafts do not advance the
committed prefix. Accepted target hidden rows rebuild raw MTP K, including
checkpoint rejection, full acceptance, forced tokens and single-token fallback.
Raw authority is mandatory even when optional accepted-prefix rebuilding is off.

The draft metadata uploader previously required nonempty *prefill entries* even
though joint drafts construct only ragged metadata. It returned early on every
joint draft. The uploader now derives the batch size from `q_indptr` and validates
the metadata shape. Tests must observe draft **writeback**, not merely draft
eligibility or a start trace. The previous continuous-sampling stage's MTP passes
therefore qualify ragged target verification with individual draft execution;
they were not evidence of joint draft kernels.

Actually enabling joint drafts exposed another transition bug: after a ragged
verify, a request falling back to a private single-request verify retained the
old shared checkpoint tensors and row indices. Restore preferred those stale
tensors, eventually raising `copy_d2d src oob` after cancellations shrank a batch.
Private verification now clears shared checkpoint references and offsets while
retaining reusable private recurrent/conv allocations.

The failed `core01` run records premature scheduler preallocation. `core03`
passes MTP KVMem with individual drafts. `core04` records four successful normal
CPU/GPU repetitions with actual joint drafts, followed by the cancellation
checkpoint failure. These historical runs remain visible and are not promoted
to final joint-execution qualification.

## Validation boundary

The model probe uses Q8_0/FP8, history lengths 768–816 plus query tokens, a
256-token working budget, 256-token generation reserve, block size 32, GPU
indexing and step updates. Query-conditioned preparation and prefix caches are
explicitly disabled to isolate page ownership and sampling. Four passes compare
CPU, CPU, GPU, GPU, reverse submission order, then repeat with selected requests
cancelled after eleven callbacks. Output lengths are 32–56 tokens.

This stage does not qualify complete query-conditioned CB preparation, refresh
beyond the generation reserve, bounded repository CB, prefix/archive reuse, or
the required long-history quality/performance matrix. Existing CB uses FP32
intermediate activations; the serial Q8 default's internal BF16 arithmetic needs
separate numerical/quality comparison. DFlash request features, paired state and
its joint proposal/acceptance scheduler remain outstanding.

No throughput claim is made from these probes. They include per-request pinned
I/O scratch initialization, trace logging, heterogeneous sampling and cancellation.
Formal performance needs the frozen protocol and independent alternating runs.
In this probe configuration, the existing I/O pools allocate four prefill,
two stage-out and two read slabs of 128 MiB each: 1 GiB of pinned scratch per
request, separate from the 256 MiB CPU tier. The CPU tier already has a shared
reuse pool; the I/O scratch still incurs repeated setup. Reusing that scratch
after all DMA/worker activity drains is a concrete remaining latency optimization.

## Closed concurrent results

`q8_shared_pool_summary01.json` verifies **544 responses**, including **384
KVMem responses** and 272 GPU-sampled responses. All 6,030 GPU decisions have
zero compatibility fallbacks (96,480 bytes of sampling decision D2H, excluding
other metadata/argmax traffic). MTP traces contain 3,124 joint draft writebacks
and 808 actual ragged target verification batches, reaching 4/16 requests.

| Case | Responses | Closed record |
| --- | ---: | --- |
| KVMem none, C4 | 32 | `core02` |
| KVMem MTP, C4 | 32 | `core05` |
| KVMem none, C16 | 128 | `core05` |
| KVMem MTP, C16 | 128 | `core05` |
| Dense MTP, C4/C16 | 160 | `core05` |
| KVMem MTP, C4, snapshot replay | 32 | `core05` |
| KVMem MTP, C4, optional prefix rebuild off | 32 | `core05` |

The 33 host CTests pass after the final checkpoint correction
(`q8_shared_pool_host_ctest03.log`). No new kernel sanitizer claim is made by
this stage; the previous operator memcheck/racecheck records retain their scope.
The failing `core04` device-copy bounds check prevented an out-of-range CUDA
copy; the same cancellation case now passes with stale shared references cleared.

The final single-request regression also passes **24 CPU/GPU pairs** (48
responses, 64 output callbacks each): eight each for none, MTP and DFlash.
These use 4K history, CPU indexing and the complete existing query-conditioned
single-request controller. Its immutable FP8 remapping remains enabled. The
closed record is `q8_gpu_sampling_validation_shared_pool_regression01.json`.
It establishes regression coverage for that controller after the common
executor/checkpoint edits; it does not qualify the advanced CB controller.

`implementation_validation.json` freezes this stage's source, binary and log
identities, including explicit failed attempts. The previous snapshot is retained
as `implementation_validation_pre_shared_pool.json`. The public performance gate
remains unrun and the goal remains active.

Reproduce with a new run ID (the runner refuses to overwrite evidence):

```bash
python3 benchmark/dflash2/run_q8_shared_pool_validation.py --run-id NEW_ID --cases none-kvmem-c4 mtp-kvmem-c4 none-kvmem-c16 mtp-kvmem-c16 mtp-dense-c4 mtp-dense-c16 mtp-snapshot-kvmem-c4 mtp-mandatory-raw-kvmem-c4
python3 benchmark/dflash2/run_q8_gpu_sampling_validation.py --run-id NEW_ID --cases dflash mtp none
ctest --test-dir build-dflash-host --output-on-failure -j 4
```
