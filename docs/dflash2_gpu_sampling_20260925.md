# Q8/FP8 DFlash 2 GPU sampling and cache remapping — 2026-09-25

The persistent goal remains **active and incomplete**. This stage moves target
sampling and speculative acceptance/residual draws onto CUDA for single-request
native generation. Target weights remain Q8_0 and target KV remains E4M3 FP8.
It adds no BF16 target loading or KV support.

## FP8 cache correctness discovered during timing

The initial warm sampling benchmark failed at zero-based callback 74. A separate
eight-request CPU-only audit reproduced differences after requests 1, 4 and 5
(zero-based), so switching samplers was not the cause. The existing default
allowed eight in-place delta-RoPE writes before rebuilding a resident block.
FP8 rounding accumulated across those moves even with immutable raw K enabled.

An otherwise identical CPU-only experiment set
`QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1`; all eight cold/warm requests then produced
the same 256 callbacks. Native FP8 now uses this one-token absolute-delta
threshold by default: every changed position reconstructs K from immutable raw
authority, while unchanged resident slots still skip work. Explicit experiment
overrides remain available. A 100-move host test checks fresh raw reconstruction,
zero drift counters and skipped reconstruction on unchanged positions.

All three routes now pass the strict eight-request cold/warm test under the new
default: 24 requests, each with 256 callbacks, and no output changes.
The original and controlled experiments are
`q8_gpu_sampling_validation_cache01.json` and
`q8_gpu_sampling_validation_cache02.json`; their audit exit status indicates
successful observation, not a correctness gate. The final probe makes any
cold/warm output change an error, and `final03` runs it on all three routes with
the new default and no refresh override. Earlier FP8 lifecycle/archive evidence
retains its old reconstruction policy; required long-history/archive gates need
confirmation on this updated numerical policy before final acceptance.

## Implementation and compatibility

`TargetSamplerCuda` builds the target distribution directly from device logits,
including presence/repetition penalties and hypothetical accepted draft tokens.
It supports the existing joint and top-k-first filtering policies, temperature,
top-k, top-p, min-p and greedy penalties. The common top-k <= 31 route uses the
existing draft selector reduction; larger candidate sets use segmented radix
sorting. Unfiltered sampling uses parallel softmax without vocabulary sorting.

The device checks normalized target/proposal probabilities, accepts the prefix,
and samples the positive residual or the full-accept bonus. DFlash uses its
actual sparse proposal probabilities; built-in MTP uses its point-mass proposal.
The host receives a 16-byte decision instead of all verification logit rows.
This describes sampling traffic only; executor argmaxes and other transfers are
additional. Test-only probability readback is disabled in serving.

The CPU's partial-sort tie behavior cannot be reproduced by simply choosing
stable token IDs. Ties crossing a truncation boundary, non-finite input and
numerically ambiguous probability/cumulative thresholds request the existing
CPU path. Only rows reached before the first rejection affect this fallback.
The original RNG is advanced by the returned number of draws only after a
successful GPU decision. Fallback consumes no request RNG and retains the CPU's
singleton, rejection and bonus draw semantics.

GPU sampling is enabled by default on CUDA. `QW3_GPU_TARGET_SAMPLING=0` selects
the original CPU path for comparisons. Ordinary native sampling and batched
MTP/DFlash verification use the bridge. Pure greedy sampling without penalties
retains its existing argmax route. Sequential MTP verification and continuous
batching sampling have not been moved by this stage; actual DFlash continuous
batching remains unimplemented.

## Validation

- The initial real-model matrix passes 24 CPU/GPU same-route pairs: eight cases
  each for KVMem none/MTP/DFlash. It covers both filter orders, two seeds,
  penalties, full-vocabulary nucleus, unfiltered sampling, large top-k and greedy
  penalties. Each response has 64 tokens; callback pieces and complete text
  match exactly within each pair. It does not assert that different speculation
  algorithms consume RNG identically.
- After the FP8 relocation correction, all 24 CPU/GPU pairs pass again on the
  new default. Normal and bounded KVMem each pass 32 production checks;
  ordinary QW3 passes 44 checks (its dense path is unaffected by relocation).
- The final operator suite passes 1,215 cases, including 60 pinned SGLang
  distribution fixtures. The matrix reaches the real 248,320-token vocabulary,
  verifier widths 1/4/8 and every rejection boundary for proposal lengths 1–7.
  Maximum probability error versus the CPU oracle is 5.96046e-8. Fifteen cases
  explicitly use compatibility fallback; 27 ambiguous unused rows are reported
  separately and are never treated as valid reached decisions.
- An independent 6,000-trial sparse rejection oracle with target probabilities
  (0.5, 0.3, 0.2) produces counts (2996, 1790, 1214); acceptance is 0.801333
  against expected 0.8. All 6,000 decisions run on GPU. These analytic tests do
  not replace whole-model task-quality evaluation.
- Full matrix memcheck and bounded racecheck pass before and after removing the
  unfiltered sorting bottleneck, with zero errors and hazards. Final racecheck
  covers 147 cases at vocabulary 37 / width 8; it is not a full-vocabulary race
  run. All three actual model routes again pass unfiltered fixed-seed equality.
- The host suite passes all 33 tests after the new interface, serving bridge and
  FP8 remapping default, including the 100-move raw-authority test.

Primary per-run records are
`benchmark/dflash2/q8_gpu_sampling_validation_core01.json` and
`q8_gpu_sampling_validation_final01.json`. The latter passes operator, unfiltered
model and sanitizer checks, but its timing run fails and is **not a passed
pipeline**. Subsequent cache audit and timing use
`q8_gpu_sampling_validation_final02.json`; its longer warm timing also failed
when the existing cache drift recurred. `final03` validates the new FP8 default.
These preserve source and actual
binary hashes for each stage. The first build's missing CUDA constant include
and the first test's incorrect assertion on an unused ambiguous row remain in
the attempt logs. They are not counted as passing evidence.

## Controlled performance diagnostic

The first implementation exposed an unfiltered regression: full-vocabulary
sorting and serial probability traversal increased a 64-token none request from
4.924 to 8.114 seconds. The parallel softmax path removes that work. The final
single-pair unfiltered checks take CPU/GPU 5.239/4.821 seconds (none),
4.156/3.832 (MTP), and 4.129/3.698 (DFlash). These include the 4K prefill and are
diagnostic observations, not independent repeated timing.

Warm timing uses one load per route, an unmeasured CPU request to establish the
query checkpoint, one warmup pair, then three pairs with
alternating CPU/GPU order. The engineering-log history has 4096 tokens, full
prompt 4126, working set 1024, generation reserve 512 and block size 32; the CPU
tier is 4 GiB and concurrency is one. Requests generate exactly 256 tokens with
temperature 1, top-k 20, top-p .95, top-k-first and seed 101. Cache reuse is
enabled. Request time includes selection/replay and generation, excludes model
loading, and includes prompt work performed on each request.

All 12 warm CPU/GPU pairs (including three warmup pairs) pass exact callback and
text equality. End-to-end medians over three measured pairs per route are:

| Route | CPU sampling tok/s | GPU sampling tok/s | Median paired speedup |
| --- | ---: | ---: | ---: |
| None | 39.496 | 43.744 | 1.114x |
| Built-in MTP | 85.880 | 113.892 | 1.325x |
| DFlash 2 | 78.965 | 112.903 | 1.426x |

GPU decode medians from rounded native seconds are 44.889, 122.959 and 121.500
tok/s, respectively. DFlash sampling-phase median drops from 0.952 to 0.012
seconds per 256-token response; verification still costs 1.722 seconds and draft
generation 0.342 seconds. MTP sampling drops from 0.695 to 0.013 seconds.
All warm GPU requests report zero compatibility fallbacks. Sampling-decision
D2H totals are 4096 bytes (none), 1120 (MTP), and 912 (DFlash), per response;
other executor transfers are additional.

The summary script preserves every request and excludes only the explicitly
designated warmup pair. Native phase seconds are rounded to milliseconds. The
MTP `sample_s` field in the initial binaries was not instrumented; its zero must
not be interpreted as zero sampling cost. The final02 binary times acceptance
and sampling for both MTP and DFlash.

Both failed timing attempts are retained. The four-request audit initially
matched warm CPU -> GPU -> CPU, but a longer repetition exposed later drift.
Priming alone was insufficient. Current comparisons use the corrected FP8
reconstruction policy and still establish equal cache conditions before timing.

This is one process per route, not the goal's three independent runs. There is
no concurrency, peak-memory, TTFT or task-quality qualification here. The sampled
outputs differ across algorithms, so CPU/GPU speedup within each route measures
this optimization; cross-route throughput is not a quality-qualified speedup.
In particular, the 112.903 / 113.892 DFlash/MTP observation does not establish
that DFlash outperforms the strongest baseline. Formal performance remains open.
The separate greedy diagnostic repeats the earlier 833-token history / 848-token
prompt, 256-token working set and generation reserve, 32-token blocks and 256
output tokens. Under the corrected FP8 policy, every none/DFlash output matches.
Three warm requests give end-to-end medians **44.933 / 85.617 tok/s** (ranges
44.642–45.206 and 85.553–86.125), respectively. This is 1.905x on this diagnostic,
with none first and one process per route. The earlier ~44/88 results used the
old remapping policy and remain historical. No public performance gate has passed.

## Reproduction and remaining work

```sh
cmake --build build-dflash-flashinfer -j 4 --target qw3 qw3-gpu-target-sampling qw3-gpu-sampling-backend-probe qw3-dflash-backend-probe
python3 benchmark/dflash2/run_q8_gpu_sampling_validation.py --run-id UNIQUE --cases upstream memcheck racecheck cache-all none mtp dflash kvmem kvmem-bounded greedy-timing timing
python3 benchmark/dflash2/summarize_q8_gpu_sampling.py --run-id UNIQUE
```

The runner removes inherited QW3 settings, selects FP8 KV, executes GPU jobs
sequentially, refuses overwriting evidence and records closed log hashes.
`q8_gpu_sampling_summary_final03.json` contains all 72 same-route sampling
responses, 12 greedy diagnostic records and their phase data. The final target
GGUF and both draft files were rehashed after testing and match the frozen
manifests (`q8_gpu_sampling_model_identity_01.json`).
`public_execution_protocol_v7.json` freezes canonical FP8 relocation and GPU
sampling for all native performance contestants; dataset and acceptance
thresholds are unchanged.
Whole-model stochastic quality, actual DFlash continuous batching, advanced
KVMem HTTP/bounded lifecycle and soak tests, qualified public reference and
formal independent performance comparisons, and final workspace integration
remain open.
