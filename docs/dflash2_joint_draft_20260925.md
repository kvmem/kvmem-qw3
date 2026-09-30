# Joint DFlash draft execution — 2026-09-25

The Q8_0 target / E4M3 FP8 KV goal remains active. This stage passes its scoped
correctness matrix; no final performance or production-lifecycle gate is claimed.

## Implementation

`DFlashBatchExecutor` groups requests by proposal width and temperature. Its
five-layer forward jointly executes the draft projections, grouped dynamic
convolutions, RMS normalization, non-causal sliding attention, FFN and residual
operations. Draft linear arithmetic now retains separate request GEMM shapes
while converting the concatenated activation once. It then jointly evaluates the shared target head, top-k and the
conditional selector. The Q8 target head uses the existing stable verification
math; official BF16 draft weights remain unchanged. There is no new BF16 target
loading or BF16 KV support.

Each request retains its own committed draft K/V, raw K, logical/RoPE positions,
noise block, RNG and proposal. Attention uses a 3-D grid and a caller-owned
metadata table of request cache pointers and key counts. The metadata uploader
synchronizes host lifetime with the device stream. Query/key RoPE and cache-tail
copies are per-request; cache ingestion from committed target features also
remains per-request. The joint forward writes only temporary noise K/V beyond
each committed context and publishes the final hidden block back to its owner.

The existing cold KVMem controller now yields a draft event before target
verification. The scheduler groups and executes proposals, writes the result
into each paused controller and resumes the original acceptance/rollback flow.
The feature is opt-in with `QW3_CONTINUOUS_DFLASH_BATCHED_DRAFT=1` alongside the
existing DFlash/continuous-batching flags.

## Rendezvous correction

The first scheduler implementation serviced draft and verification events in
the same iteration. When a queued request filled a freed slot, it could remain
one phase behind its peers, repeatedly splitting target batches. The initial
C4 test passed 32 responses. The subsequent replenishment run had passing
completed repeats but was deliberately terminated to fix this scheduling issue;
`model02` is cancelled evidence, not a failed correctness qualification.

The corrected scheduler retains ready verification events while other resident
controllers finish drafting. It starts target verification only when every live
controller is ready. New admission occurs between target rounds so queued
arrivals cannot indefinitely delay a held verifier. Borrowed logits from the
previous target batch are consumed before any new draft batch executes.

## Evidence so far

`operator01` passes nine independent-cache attention configurations with 1/4/16
requests, query widths 1/3/8 and cache lengths up to 8192. Reversing requests
preserves bit-exact agreement with private attention and output guards.

The frozen official FP32 and BF16 draft fixtures each pass 252 joint-request
checks: widths 1/3/8, request counts 1/4/16, both request orders, greedy and
stochastic proposals. Greedy tokens match the topology oracle. Joint selection
agrees with private selection on each published hidden block and consumes the
same per-request RNG draws; maximum probability error is 2.683e-7. Committed
archive payloads remain unchanged after every proposal. These are small-model
numerical fixtures, not actual-checkpoint throughput or quality benchmarks.

The pre-rendezvous real Q8/FP8 C4 test passes all 32 CPU/CPU/GPU/GPU normal and
cancelled responses, with actual joint projections, convolution, attention and
selection visible in the trace. Wider/replenishment/refresh regressions,
post-rendezvous host checks and CUDA sanitizer checks remain in progress.

The first post-rendezvous replenishment run (`model03`) fails the final normal
GPU repeat for request 0 at zero-based output callback 16. Earlier CPU repeats
and the first GPU repeat pass. This is a failing integration gate; the feature
remains opt-in. Added proposal fingerprints and batch-width tracing will
distinguish draft arithmetic drift from controller state errors. No cause is
asserted yet. Both pre/post-rendezvous host builds pass all 34 host tests, which
does not override the failing real-model result.

## Batch-shape diagnosis and correction

The instrumented `trace01` also fails, this time in the second CPU repeat.
For seed 101 its first proposal already has a different probability fingerprint
when the draft group changes from four requests to one; the proposed token IDs
are initially identical. The otherwise identical `control01`, with joint draft
disabled, passes all 64 responses and has identical proposal fingerprints in
all four normal repeats. The closed comparison is
`q8_joint_draft_trace_comparison01.json`.

The new `qw3-dflash-linear-batch` probe isolates GEMM from all scheduler, target,
cache and sampling state. It uses the actual official draft weights for seven
projection shapes, widths 1/3/7/8 and 1/2/3/4/16 requests. Identical FP32 input
rows are evaluated privately and as one concatenated GEMM. `linear01` fails
bitwise equality in 71 of 140 cases, with maximum absolute difference
0.000732422. This establishes batch-dependent draft GEMM arithmetic; passing
the small-model topology tolerance did not guarantee fixed-seed reproducibility.

The correction adds `dflash_linear_blocks`: CUDA converts the concatenated
input to BF16 once and issues the same fixed-size GEMM for each request.
Projection width remains the request's block size (one less for the selector),
independent of admission, cancellation, group order and number of peers. Other
backends use equivalent independent tensor views. This retains the private
draft arithmetic and reduces activation conversion launches, but does **not**
claim a single shared GEMM or a measured performance win. A faster joint GEMM
must satisfy this arithmetic reference before replacing it. No target precision
or model-loading behavior changes.

Post-correction `stable01` passes all 140 same-input GEMM cases bit-exactly,
the CUDA operators and both 252-request FP32/BF16 topology fixtures. Host CTest
passes 34/34. Its replenishment case passes 64 responses, alternating private
and joint drafts across CPU/GPU sampling and callback cancellation. All 495
proposal comparisons have identical candidate/probability fingerprints and
token IDs, in addition to exact final output. The feature remains opt-in and
performance is unqualified. The same final linear-probe binary reproduces
71 failing cases with the old concatenated GEMM (`negative01`) and passes all
140 with fixed request shapes (`stable01`). Both records retain the same runtime
and test source hashes; failure evidence is preserved.

The complete `stable01` matrix passes **352 responses**, including **320 KVMem**:
64 replenishment/private-joint parity, 128 at C16, and 32 each for external
cancellation, greedy verification, completed guided queries, repeated refresh
and dense QW3. Logs show 1,997 joint draft groups, 1,632 ragged verification
batches, 320 query replays, 32 completed guided queries and 84 actual refreshes
(four with generated queries; the others exercise the existing fallback policy).
GPU target sampling records 3,098 calls and zero fallbacks. CUDA operator
memcheck reports zero errors; racecheck reports zero hazards, errors or warnings.
These sanitizer results cover the operator suite, not every host concurrency
interleaving or the entire model process.

The matrix uses cold requests and the existing CB FP32 intermediate arithmetic.
Shared prefix reuse, archive/session/compaction, bounded repositories,
serial-versus-CB task quality, the frozen formal matrix and public comparison
remain separate unclosed gates. The closed evidence is
`q8_joint_draft_summary01.json` and its per-run manifests.

## Diagnostic timing limits

The replenishment parity run compares private and joint proposals in four
CPU/GPU/normal/cancel pairs. The ratio of private time to joint time is
0.959–0.996: this cold, short-output probe does not demonstrate an end-to-end
improvement. Width-eight joint proposal median times are 5.090/8.440/11.876/
15.410 ms for groups of 1/2/3/4 requests. Group size and admission order vary,
so these phase timings are not a fixed-work throughput benchmark.

The remaining optimization work includes a genuinely shared projection GEMM
that retains the arithmetic reference, joint prefill and controlled end-to-end
measurements. Current projection GEMMs remain per-request even though input
conversion and the other draft operators execute jointly.

## Reproduction

Use a fresh run ID; the runner refuses to overwrite earlier evidence. GPU
jobs must run sequentially.

```sh
python3 benchmark/dflash2/run_q8_joint_draft_validation.py --run-id NEW_ID --cases linear-batch operators forward-fp32 forward-bf16 replenish-parity-c4 kvmem-c16 external-cancel-c4 greedy-c4 guided-c4 refresh-c4 dense-c4 memcheck racecheck
```

`linear-generic` intentionally checks the old concatenated GEMM and is expected
to fail exact equality on the recorded CUDA/cuBLAS environment. It is a negative
arithmetic control, not an enabled production path. `linear01` was recorded
before the probe acquired this explicit mode switch, and therefore retains its
original `linear-batch` label and exact source/binary identities.

The continuous probe's `draft_parity` and `joint_draft` output fields describe
the alternating parity experiment only. When `draft_parity=0`, actual joint
execution is established by the `continuous_dflash_draft` operator trace and
the per-run environment, rather than that parity-only flag.
