# FP8 KVMem support and baseline timing audit

Date: 2026-09-24. Goal active; KVMem is the primary performance deliverable.
Implementation: `/home/chaidi/qw3-dflash2`.

**Precision scope update:** the user subsequently required Q8_0 target weights
with FP8 KV and removal of the new BF16 target/KV code. BF16-target runs below
are historical diagnostics only. Their artifacts are retained to explain the
baseline correction; they do not establish current Q8_0 DFlash qualification.

## Precision and the reported ~21 tok/s baseline

The earlier short diagnostic used **BF16 target weights and FP8 KV**. It did
not use the existing Qwen3.8-27B-Q8_0 deployment. The backend loads 51,300 MiB
of target tensors for BF16 versus 27,260.6 MiB for Q8_0. FP8 KV changes cached
keys/values, not weight precision or the amount of model-weight data read.

The user's existing ~40 tok/s plain baseline is consistent with a fresh run of
the current binary using Q8_0 weights, FP8 KV and no MTP: the same KVMem
regression fixture reports 47.25 tok/s for its all-fit prompt, 34.68 tok/s for
its first sparse query and 43.05 tok/s for its repeated sparse query. These are
32-token diagnostics, not a repeated public benchmark. Raw evidence:
`benchmark/dflash2/q8_fp8_baseline_audit_01.log` and its run manifest. The plain
cases pass, but the subsequent DFlash identity case fails serial output parity;
the run's nonzero return code is retained and must not be counted as a pass.

A second issue affected the interpretation of the short rates. The old plain
timer measured from initial prefill end through working-set selection, query
replay and decode. The MTP/DFlash timer already starts after post-prefill work.
Thus the old **84.92 / 21.07 must not be reported as a decode speedup**.
The end-to-end observations (32 / 1.82344 = 17.55 tok/s versus
32 / 2.77338 = 11.54 tok/s) still describe those two individual BF16-weight
requests, including a first sparse-query cache miss. They are not a Q8_0
deployment result, a warm steady-state result, or a public benchmark.

The plain route now separates `postprefill` from `decode`, adds
`timing_version=2`, and preserves the old duration as
`legacy_decode_including_postprefill`. This is a measurement correction, not
an inference optimization. The production probe additionally records TTFT and
the interval between its first/last token callbacks. For the ordinary route,
`(tokens-1)/callback_span_s` independently checks steady generation speed.
DFlash callbacks can arrive in bursts; their intervals are not kernel timers.

The longer audit generates 256 tokens per request, followed by three repeats
of the same sparse query in one loaded process. Its script and results are
`benchmark/dflash2/run_baseline_timing_audit.py` and
`benchmark/dflash2/baseline_timing_audit_01.json`. It fixes history (833 tokens),
query (848 total), working set (256), generation reserve (256), block size (32),
FP8 KV, greedy sampling, seed 101, single request, and no MTP/DFlash. Only target
weight precision changes. Three warm requests are not three independent runs.

The completed 256-token plain audit gives these median warm-request rates
(three requests, range in parentheses), on the same binary and GPU:

| Target weights / target KV | Native decode tok/s | Callback decode tok/s | End-to-end tok/s |
| --- | ---: | ---: | ---: |
| Q8_0 / FP8 | 45.61 (45.32–45.86) | 45.44 (45.16–45.70) | 44.95 (44.68–45.18) |
| BF16 / FP8 | 24.32 (24.31–24.43) | 24.23 (24.22–24.34) | 24.08 (24.07–24.18) |

The first sparse query's native decode / end-to-end rates are 44.24 / 36.42
tok/s for Q8_0 and 24.44 / 21.40 for BF16. Cold/warm effects particularly
change end-to-end rates; they should not be mixed. Both long plain audits pass
their callback-count and EOS-limit checks. No MTP or DFlash weights are loaded.
This reproduces the user's ~40 tok/s magnitude on the current implementation;
it is not a claim to reproduce their unidentified historical prompt exactly.

After the timing correction, the BF16-weight/FP8-KV KVMem production regression
passes all 32 plain/DFlash checks. Across the two long audits and that regression,
all 25 ordinary generated requests satisfy the phase accounting identity
(`postprefill + decode = legacy combined duration`, within printed rounding),
and first/last callback timestamps stay within the independently measured
request wall time. The rebuilt binaries and completed logs are recorded in
`implementation_validation.json`; historical pre-fix manifests remain available.

## FP8 support that has passed

With BF16 target/draft weights and E4M3 FP8 target KV, the production probe
passes dense 44, normal KVMem 32 and bounded-repository KVMem 32 checks. These
cover same-target greedy output, both sampling orders, fixed-seed repeats,
penalties, EOS, cancellation/retry, sparse query replay, warm prefix restoration
and tail rewrite. `fp8_stage_backend_runs_01.json` records the binary SHA, argv,
environment and return codes. This matrix ran before the contiguous gate fix;
its default paged route does not use the corrected contiguous fallback.

The contiguous FP8 small-batch fallback applied the sigmoid attention gate
inside its nominally ungated operation, then the gated wrapper applied it
again. The extra multiplication was removed. An independent analytic oracle
uses zero keys, known constant values, varying gates and guarded output rows:
24 cases cover history lengths 65/8193, batches 1/2/7/8/9/33, plain/gated calls.
Before the fix it fails (max absolute error 1.09964); after the fix all cases
pass (maximum 1.19209e-7) and Compute Sanitizer reports zero errors. Original
failure and successful logs/manifests are retained as `fp8_gate_*` artifacts.

The real-model contiguous FP8 follow-up (math prompt, 64 tokens) passes greedy
output parity, rollback after keeping 1..7 rows, exact continuation logits and
paired target/draft snapshots. See `fp8_contiguous_target_run_01.json` and log.
This single prompt does not establish general serial/batch numerical parity.

## Remaining KVMem work

Q8_0 weights plus FP8 KV still need DFlash output parity qualification. The
new identity-prompt failure joins the earlier Q8 Chinese-prompt gate; its exact
cause has not yet been isolated. BF16 KV also has an open Chinese parity gate.
BF16-weight/FP8-KV fixture passes do not close those separate configurations.

Long history, archive restore, guided/semantic replay, compaction, repository
exhaustion/GC, advanced sessions and quality versus higher-precision KV remain
unqualified. DFlash continuous batching and GPU target sampling remain open.
The existing SGLang reference qualification uses BF16 KV; an FP8 comparison
requires a separately qualified matched precision configuration. No public
performance gate has passed. Ordinary QW3 is a correctness/regression control;
the performance target and next implementation work are KVMem.
