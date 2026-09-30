# DFlash continuous target verification with the KVMem controller — 2026-09-25

The persistent Q8_0 target / E4M3 FP8 KV goal remains **active and incomplete**.
This stage connects DFlash to the shared target verifier while retaining the
existing complete per-request KVMem control flow. Draft proposals still run
individually. Shared draft linear/attention execution, full production lifecycle
qualification and formal performance gates remain outstanding.

## Scheduling and request ownership

The existing `generate_mtp` controller already owns DFlash preparation,
query-conditioned retrieval/replay, private queries, mid-decode refresh,
acceptance, rollback, thinking boundaries and cancellation. Reimplementing those
operations in the older simple CB loop would create two independent policies.
`CooperativeRequest` instead preserves each controller's C++17 stack on a host
thread. The scheduler resumes exactly one controller, waits until it yields or
finishes, then resumes another or executes a joint GPU verification batch.
Controllers do not access the shared CUDA backend simultaneously.

The yield occurs at ordinary target verification. Private query operations and
transactional replay keep using the full controller. Each request owns its
executor, draft cache, RNG, sampler and transaction state. Fixed scheduler slots
keep paused references stable. Completion or error joins/unwinds the controller
before destroying its executor or returning shared KV pages. Requests that fail
admission or are cancelled before initialization still receive a completion.
The controller callback checks the request cancellation flag before and after
the client callback, and the verification yield checks it before submitting a
batch. Thus external cancellation remains observable even when the client
callback itself returns true.

The feature is currently opt-in through `QW3_CONTINUOUS_DFLASH=1`, alongside the
existing engine/request continuous-batching flags. Sampling batches also require
`QW3_CONTINUOUS_BATCHING_SAMPLING=1`; otherwise the complete request executes
without a joint verification yield. Tests inspect actual joint executor traces.

## Feature, checkpoint and sampling transport

Ragged target verification allocates each request's feature view and captures
the DFlash taps after their target layers (official taps 5/19/33/47/61). At the
end it publishes logical and compact RoPE positions and the valid row count.
Checkpoint rejection clamps that count before committed features enter the
request's draft context. The separate ordinary body-decode path still excludes
DFlash feature capture; single-token DFlash transitions use their complete
executor, which already captures features.

The controller receives its own shared checkpoint rows and target argmaxes.
GPU sampling borrows the joint logits matrix with an explicit row offset and
uses the original request's DFlash sparse proposal and RNG. CPU sampling owns
the corresponding rows. All controllers consume their decision/checkpoint data
before the next joint target batch overwrites the shared buffers. Optional
last-row publication copies into the executor's existing logits buffer.

The first real KVMem run exposed an implicit shared-executor dependency:
`kvmem_replay_selected_prefix` inspected the backend's main executor rather than
the current controller's override executor. Its selection did not match the
request's live window, so query replay correctly rejected it. The helper now
requires the executor explicitly. Failed `core01` evidence is retained.

## Validation scope

The concurrent DFlash probe enables CPU content indexing, query-conditioned
selection and forced query replay. It uses 768–816 history tokens plus a query,
a 256-token working budget, a 256-token generation reserve and a 256 MiB CPU
tier. It compares CPU/CPU/GPU/GPU repeats with reversed admission order and
selected requests cancelled after eleven callbacks. Each request produces
32–56 output callbacks in the short cases when not cancelled. The completed
mid-decode-query case uses a 1024-token reserve and 96 outputs so that a safe
text boundary can generate a private query. The long case retains the 256-token
reserve and emits 320–344 tokens, refreshing at 96-token epochs. Dense QW3 runs
provide regressions. Additional cases compare greedy individual/joint target
verification, enqueue eight requests into four slots, and cancel via the request
flag while the client callback keeps returning true.

The cooperative host test covers 400 alternating yields with four requests,
preserved stack locals and replies, serialized execution, cancellation/unwind,
destruction before first admission and exception transport. AddressSanitizer and
UndefinedBehaviorSanitizer pass that test. ThreadSanitizer could not initialize
in this environment: both PIE and non-PIE attempts fail with an unexpected
memory mapping. They are unavailable checks, not passing race checks.

The final `core05` default matrix passes **448 responses**, of which **416 use
KVMem**. Every case compares CPU/CPU/GPU/GPU repeats both normally and with
cancellation. The 384 DFlash responses comprise 352 KVMem and 32 dense responses;
the none/MTP front-end controls contribute another 64 KVMem responses.

| Final case | Responses | Required behavior |
| --- | ---: | --- |
| Greedy individual/joint verification, C4 | 32 | Same outputs inside the CB FP32 configuration |
| Eight requests replenishing four slots | 64 | Slot reuse while other requests remain resident |
| External cancellation, C4 | 32 | Flag is honored while the callback returns true |
| KVMem C16 | 128 | Actual joint kernel batch reaches 16 |
| Dense C4 | 32 | Ordinary QW3 regression |
| Completed initial private query, C4 | 32 | All 32 private queries complete directly |
| Completed refresh query, C4 | 32 | All 28 required refreshes generate a private query |
| Repeated refresh, C4 | 32 | Outputs up to 344 tokens cross three refresh epochs |
| None and MTP KVMem C4 | 64 | Existing front-end controls remain unchanged |

The traces contain 1,611 multi-request DFlash verification batches and 225
single-request ragged tails. DFlash performs 352 query replays and 112 refreshes.
The final matrix enables GPU sampling for 224 responses; 212 require sampler
calls (the greedy parity case mostly bypasses distribution work). Its 4,462 GPU
sampling calls use 71,392 decision-readback bytes and zero CPU fallbacks.
These counts are checked from closed, hashed logs by
`benchmark/dflash2/summarize_q8_dflash_cb.py`, with results in
`q8_dflash_cb_summary01.json`.

Earlier `core02`/`core03` results retain their original binary identities and
are excluded from these final totals. Failed intermediate runs are retained.
The host suite passes all 34 tests. Single-request none/MTP/DFlash regression
also passes all 24 CPU/GPU pairs (48 responses), recorded in
`q8_gpu_sampling_validation_dflash_cb_regression01.json`. These are separate
from the 448 continuous-batching responses above: this stage's final model
regression totals 496 responses, without counting earlier experimental runs.

## Batch-shape arithmetic failure and correction

The `core04` guided-complete, refresh-complete and refresh-long cases pass, but
its eight-request/four-slot replenishment case fails CPU-versus-CPU repetition
with cancellation (request 6, zero-based output callback 37). A separate
`replenish_trace01` run reproduces a CPU repeat difference even without
cancellation (request 3, callback 51). Earlier C4/C16 groups had no queued
replacement requests and do not cover this failure.

Per-seed token tracing shows identical early proposals and acceptance, followed
by divergence after a request moves from ragged verification to its private
verifier. The paths use different Q8 batch arithmetic and attention kernels.
This is a localization result, not yet proof that no state-lifetime issue exists.
The `replenish_ragged01` experiment keeps the single-request tail on the ragged
verifier. Its normal four repeats pass, but its cancellation CPU repeat still
fails (request 7, callback 54), so that change alone is insufficient. A further
opt-in experiment tiles Q8 verification linears into decode-compatible widths
up to eight; it does not change the draft or target/KV precision. The expanded
operator oracle checks widths 9/16/31/128 with contiguous and guarded strides.
The operator oracle passes all 48 linear cases (24 wide/tiled) bit-exactly,
and CUDA memcheck reports zero errors. `replenish_stable01` then passes all 64
model responses, including CPU/GPU repeats and cancellation. The `replenish_control01` control retains front-end locking and single-request
ragged verification but disables tiled Q8. It reproduces a CPU repeat failure
(request 3, callback 51), establishing that locking alone does not resolve the
batch-dependent arithmetic. The DFlash CB default now retains ragged verification
when one request remains and enables tiled Q8. The closed `core05` matrix above
rechecks these defaults. The negative control uses identical runtime source
hashes to the passing controlled run; it disables tiled Q8 and stops after CPU
repeats. It is evidence of the numerical failure, not a passing qualification.

## Remaining boundaries

This scheduler currently starts cold request controllers: shared KVMem prefix
reuse, persistent session/HTTP state and archive/compaction lifecycle integration
still require per-request ownership and tests. Bounded-repository CB retains its
existing rejection. Inline refresh and semantic-chunk prefill retain their original
override-executor restrictions. Independent semantic budgets and explicit
transcript/oracle spans retain the existing serial admission path, where their
request-wide setup is applied. These restrictions are not relaxed to claim
completion.

The public `generate`/`generate_internal` entry path also prepares shared API
and main-executor input metadata before CB admission. That pre-existing front-end
mutation is now protected by a request-setup mutex. The mutex is released while
waiting for a CB result and reacquired before semantic-budget/input-scope
cleanup; ordinary serialized requests keep it. Named-cache metadata access uses
the same mutex. The final rebuilt admission/replenishment probes pass with this
fix; passing output probes does not establish absence of all host data races.

Existing CB uses FP32 intermediate activations, while serial Q8 defaults to the
framework's prior internal BF16 arithmetic. Individual-versus-joint verification
inside CB therefore tests its FP32 configuration; it does not replace the serial
Q8 numerical/quality gate. No BF16 target-weight loading or BF16 KV support is
added. No speedup or public-performance gate is claimed from these correctness
probes. Per-request verification wall time includes time waiting for peers;
the separate `batch_seconds` trace measures the shared verifier execution.
The tiled Q8 path is a correctness reference that can cost more than wide GEMM.
An efficient wide verifier must preserve its activation quantization and
reduction behavior; formal speedup is still an outstanding gate.

## Reproduction

Use unique run IDs; runners refuse to overwrite earlier evidence. Build the
CUDA/FlashInfer probe targets first, then run the two commands sequentially so
their GPU work does not overlap:

```sh
python3 benchmark/dflash2/run_q8_dflash_cb_validation.py --run-id NEW_ID --cases dflash-verify-parity-kvmem-c4 dflash-replenish-kvmem-c4 dflash-external-cancel-kvmem-c4 dflash-kvmem-c16 dflash-dense-c4 dflash-guided-complete-kvmem-c4 dflash-refresh-complete-kvmem-c4 dflash-refresh-long-kvmem-c4 none-kvmem-c4 mtp-kvmem-c4
python3 benchmark/dflash2/run_q8_gpu_sampling_validation.py --run-id NEW_ID --cases dflash mtp none
```

The runner fixes FP8 KV, removes inherited `QW3_` environment overrides, and
records source/binary/log hashes. The cooperative test, ASAN/UBSAN logs,
unavailable TSAN attempts and the Q8 operator/memcheck record are separate
evidence; none of these commands constitutes the formal performance matrix.
