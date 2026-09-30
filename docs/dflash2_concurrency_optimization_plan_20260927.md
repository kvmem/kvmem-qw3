# Long-history KVMem concurrency optimization order

This is an implementation plan based on the current native cold-request measurements,
not a claim that the optimizations below have been implemented or qualified.
Keep the running `long_concurrency01` baseline queue and its binaries unchanged.
Target weights remain Q8_0, KV remains FP8, each request keeps K128K/G64K.
No BF16 target-weight support, smaller retrieval window, or 130 tok/s acceptance target
is part of this plan. GPU memory is observed; the 48 GB optimization remains deferred.

## Evidence and interpretation

Seed 1000 completed 12 requests per configuration:

| C | Cold service tok/s | Aggregate callback-window tok/s | Output tokens | Sampled GPU peak, decimal GB |
|---|---:|---:|---:|---:|
| 1 | 10.586 | 80.410 | 10798 | 53.598 |
| 2 | 12.435 | 21.637 | 12931 | 65.528 |
| 4 | 13.784 | 17.379 | 14567 | 89.329 |

Callback-window timing includes other slots' prefill and selection. C2/C4 maximum
callback gaps were 87.37/221.82 seconds. Mean verification batches at 1/2/4 requests
were approximately 33/63/123 ms. These are observations across differing live shapes,
not controlled kernel speedup measurements. Acceptance was 39.55/41.96/42.29%; the
observed collapse does not coincide with a drop in draft acceptance.

The controller in `src/qwen_native_backend.cpp::run_continuous_controlled_requests`
resumes each request until it yields, prioritizes pending prefill events when the
experimental joint path is enabled, and requires all resident requests to have ready
verification work. With joint prefill off, the existing prefill loop does not yield
between its chunks. `src/q8_verification_math.hpp` preserves stable Q8 arithmetic by
using up-to-eight-row tiles; fused SwiGLU and residual-add paths also retain this
tiling. Both observations give concrete optimization candidates, but phase profiling
is still required before assigning the 33/63/123 ms costs to particular kernels.

Seed 1001 C4 also completed (16.383 callback-window tok/s); its C2 and C1 runs remain
in the original queue as of this plan. Different output lengths mean that cold tok/s
ratios are not equal-work acceleration ratios or quality qualifications.

## 1. Remove long blocking intervals without changing prefill arithmetic

Introduce a cooperative event for **independent** prefill execution. It must be
separate from `QW3_CONTINUOUS_KVMEM_BATCHED_PREFILL`, whose historical output-parity
failure is unresolved. Keep the existing chunk size and per-request executor calls
for the first A/B, so the initial change isolates scheduling from arithmetic changes.

- Yield at a completed chunk boundary after KV/raw-K writes, recurrent state and
  DFlash feature/prefix capture are consistent. Preserve cancellation unwinding.
- Track prefill, ready-to-verify, private-query/replay/refresh, and completion as
  distinct per-slot states. Run a ready verification subset without requiring every
  resident slot to reach verification first.
- Give ready decode bounded priority, then admit a bounded amount of prefill work;
  use aging or a minimum prefill share so new requests are not starved.
- Protect shared logits/scratch lifetimes. The current barrier also protects borrowed
  verification logits: consume or retain those results before another batch overwrites
  the shared storage. Removing the barrier alone is not a correct implementation.
- Keep one owner of the CUDA execution backend. Host cooperative scheduling does not
  authorize unsynchronized concurrent access by request threads.
- After unchanged-chunk parity passes, compare 512/1024/2048-token prefill budgets
  under mixed load. Smaller chunks trade prefill efficiency for output latency;
  choose using measured throughput and p95/p99 gaps, not a fixed assumed target.
- Extend safe yields to private query generation, replay and refresh preparation.
  Keep the actual KV/recurrent-state commit atomic; do not expose half a transaction.

Acceptance: unchanged-policy C1 regression; long/short mixed requests; reverse
admission; cancel during prefill/refresh; replacement after cancellation; one slot
refreshing while peers continue. Request-owned KV, RNG, recurrent state and draft
cache must remain isolated. Scheduling-induced output changes require investigation,
not automatic reclassification as acceptable noise. Additional wait should be bounded
by scheduled chunks/atomic phases instead of an entire peer's history ingestion.
Report any remaining uninterruptible selection or I/O separately.

## 2. Make target verification benefit from multiple requests

Measure the verifier separately from mixed prefill scheduling. Add a diagnostic in
which all resident prompts have finished prefill before a common decode window;
retain the cold mixed-load measurement as the serving result. Break down Q8 linear,
FP8 attention, recurrent kernels, checkpoint copies, metadata/planning and sampling.

First low-cost candidate: independently A/B the existing `QW3_Q8_VERIFY_WIDE=1`,
`QW3_Q8_VERIFY_WIDE_TILE=32` path on the new long-history controls. Historical short
tests and microbenchmarks are useful prerequisites, not proof of a new long-workload
gain. Preserve its reference path and numerical checks. It does not yet cover all
fused operations, so investigate widening/fusing SwiGLU and residual projections
only if the phase profile justifies the work.

For FP8 ragged attention, examine per-request KV splitting, workspace sizing and
metadata/plan reuse for changing batch membership and lengths. Single-request stable
prefill and its planner optimizations are not automatically inherited by ragged
verification. The plan/cache key must reflect the actual shape and layout; stale
plans must not survive slot reuse, refresh, or cancellation.

Use fixed-input operator/verification replay to isolate speed and numerical changes,
then run real frozen-prompt end-to-end A/Bs. Keep arithmetic controls, stable request
IDs, seeds, output lengths and acceptance statistics. A faster but numerically
different path needs explicit numerical/state/quality evaluation before adoption.
Do not promise 2x or 4x: distinct requests still read distinct long KV histories.

## 3. Reuse history across real agent turns

The cold workload intentionally re-ingests 131K-263K tokens and does not represent
warm multi-turn agent service. Retain each session's prefix lease, KV window, raw
history, DeltaNet checkpoint and draft state; ingest the exact new suffix on the
next turn. Start with the same compatible session, validate prefix identity, and
keep cold and warm benchmark reports separate. Do not obtain apparent improvement
by warming only the candidate side of a cold A/B.

Session ownership, cancellation, page recycling, refresh commit/rollback and bounded
CPU/NVMe lifecycle must be implemented before relaxing the current HTTP compatibility
guards. Removing those guards is not a substitute for bounded KVMem support. This
step is necessary for a real DeepSWE agent-throughput claim and may have a larger
end-to-end effect than kernel tuning when prefixes are reusable.

## 4. Optimize draft and launch overhead after the dominant costs fall

Then profile draft linear layers, CPU launch/synchronization and small copies. The
existing joint draft still uses per-request projection GEMMs to preserve arithmetic;
past concatenated GEMMs changed results. Only pursue a new batched draft or graph
capture if the profile supports it and the same state/numerical tests pass.

## Measurement order and stopping rule

Finish the original two-seed baseline first. Make each candidate independently
switchable and use fresh frozen builds/results directories. Validate correctness on
short mixed-state cases, then C2 long history, then C4; retain C1 as a regression.
Compare the scheduler-only change before combining it with wide verification or
attention changes. Add continuous replenishment after fixed-group behavior is
understood, and a warm HTTP/session workload only after that path is implemented.

Report service wall time, output tokens, common-window decode rate, per-request
TTFT and p95/p99 gaps, actual batch distribution, draft acceptance, phase timings,
and memory. Preserve failures and record candidates with no measured benefit.
Stop pursuing candidates when their measured return is too small relative to the
remaining complexity; 130 tok/s is a reference, not a required finishing threshold.
