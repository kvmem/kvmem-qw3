# Q8/FP8 continuous sampling and shared MTP cache isolation — 2026-09-25

The persistent DFlash goal remains **active and incomplete**. This stage extends
the existing GPU target sampler to the existing none/MTP continuous-batching
scheduler. DFlash is still excluded from that scheduler until per-request draft
features, commit/rollback and the complete KVMem controller are implemented.
These changes add neither BF16 target-weight loading nor BF16 KV support.

## Implementation

Each active request owns a sampler bridge and RNG. Ragged prefill, decode and
MTP verification publish a borrowed device logits matrix plus its request row
offset. The bridge creates a bounded non-owning view, samples the request, and
advances only its RNG. Views remain alive until the decision is synchronized;
all consumers finish before another batch overwrites the matrix. The existing
numerical compatibility fallback copies only the affected request's rows.
CPU-only controls retain owned host rows. Initial tokens and single-request
fallback steps use the same bridge. Pure greedy argmax remains unchanged.

The serving decision is 16 bytes. This is sampling traffic, not total D2H;
existing argmax and metadata transfers are additional. The operator regression
also checks 24 offset-view calls against independent request copies, with NaN
sentinel rows, alternating interleaving, widths 1/4/8, point-mass and DFlash sparse
proposals, unchanged RNG on fallback and rejection of an overflowing row offset.

## Shared MTP KV defect

CPU-only repeated concurrent requests exposed output changes when admission
order changed. Disabling batched draft or batched verification independently,
and then both together, still reproduced the problem. In the external shared
MTP cache, the single-token and prefix-prime paths used dense logical addressing
unless an optional paged-prefix environment switch was enabled. These reads and
writes bypassed each request's physical page mapping and could alias other
requests. Both paths now force paged addressing whenever an external MTP cache
is attached. This requirement cannot be disabled by the optional switch.

The failed attempts remain in `q8_cb_sampling_validation_core02.json` and
`audit01`/`audit02`/`audit03`. `core01` separately records a probe setup mistake:
it omitted the request-level continuous-batching flag and entered synchronous
execution concurrently. The probe now sets both engine and request flags.

## Validation scope

The live probe concurrently admits heterogeneous prompts, seeds, filter policies
and output lengths. It compares CPU repeats and GPU outputs, reverses launch
order, cancels selected requests after eleven callbacks, and reuses the worker
for later requests. Dense and KVMem modes are separate. KVMem uses Q8_0 target
weights, FP8 KV, 768–816 history tokens, a 256-token working budget, GPU indexing,
256 MiB CPU tier and step updates. Query-conditioned preparation is explicitly
disabled in this sampler-isolation probe; advanced KVMem controller validation
is still required. The probe is not a formal throughput benchmark.

Existing CB forces FP32 intermediate activations; single-request Q8 defaults to
the framework's existing BF16 intermediate activations. Same-route CPU/GPU
agreement therefore does not qualify CB against the single-request numerical
path or demonstrate stochastic task quality. The Q8/FP8 precision label alone
is insufficient to equate those configurations.

KVMem continuous execution is **not qualified**. The corrected probe reaches
the native immutable-K guard and is rejected: the external shared target KV
arena does not install the bounded tiered page allocator required by raw-K
authority. `core04` retains that failure; `core03` also retains an earlier probe
configuration rejection (CPU indexing requires query conditioning). Neither is
counted as a passing KVMem test. Immutable raw K remains enabled.

The next allocator change must give each request bounded resident/private page
quotas over the shared arena, return pages safely on cancellation/rollback and
preserve the external MTP allocator during `configure_kvmem` (the current
teardown unconditionally detaches it). Main and MTP tiering must agree on raw-K
and spill layouts. Prefix leases, retrieval/replay, bounded repository state and
the draft's features/snapshots need corresponding per-request handling before
the DFlash route can enter the scheduler. Removing the admission guards alone
would not implement these semantics.

Allocator design also needs to separate the request quota from the valid
physical-page ID range: current `KvPageTable` validates IDs against the
allocator's `total_pages()`. A request borrowing high-numbered global pages
cannot report its smaller quota through that same range check. Tests must cover
noncontiguous/high physical IDs, quota exhaustion, borrowed prefix ownership,
configuration teardown and cancellation without double release.

Dense C4/C16 now pass for none and MTP: **320 responses**, including 160 GPU
responses. CPU repeated controls, reversed submission and cancellation pass.
The 4,346 GPU decisions have zero compatibility fallbacks. Trace records confirm
actual 4/16-row body batches and multi-request ragged MTP verification, including
zero host logit rows on the GPU path. The inherited `body_batch_fp16` trace name
also labels FP8 execution; the probe explicitly sets `QW3_KV_DTYPE=fp8` before
loading the model.

The operator suite passes 1,215 cases including 60 pinned upstream fixtures;
the additional 24 ragged-view cases pass. Full sampler memcheck has zero errors,
and the bounded racecheck has zero hazards. The 33 host CTests pass after the
external-cache fix. Single-request KVMem none/MTP/DFlash regressions each pass
eight CPU/GPU pairs (24 pairs, 48 responses, 64 output callbacks each), including
both filter orders, penalties, full-vocabulary and large-k cases. These use the
existing complete query-conditioned controller and canonical FP8 raw-K policy;
they do not enter the unsupported KVMem CB path.

Evidence is frozen in `q8_cb_sampling_summary01.json` and the per-run `core02`,
`core03`, `core05` manifests. Those manifests retain both closed successes and
explicit failures; only the named passing dense cases enter the summary. No
new throughput or public-performance qualification is claimed.

Reproduce from the worktree with the existing FlashInfer CUDA build and models:

```bash
python3 benchmark/dflash2/run_q8_cb_sampling_validation.py --run-id NEW_ID --cases operator none-dense-c4 mtp-dense-c4 none-dense-c16 mtp-dense-c16 memcheck racecheck
python3 benchmark/dflash2/run_q8_gpu_sampling_validation.py --run-id NEW_ID --cases dflash mtp none
```

Choose unused run IDs: both runners refuse to overwrite evidence. The explicit
`none-kvmem-c4` CB case reproduces the current allocator rejection. Keep that
failure visible until the shared tiered allocator and controller are complete.
