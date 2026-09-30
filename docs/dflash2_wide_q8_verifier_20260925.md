# Wide Q8 target-verification experiment — 2026-09-25

The Q8_0 target / FP8 KV goal remains active. The preceding controller stage
passes 448 continuous-batching responses and 48 single-request responses;
its frozen evidence remains separate from this optimization experiment.

## Implementation

The stable DFlash verifier previously submitted Q8 projection tiles of up to
eight activation rows from the host. Each tile quantizes the activation and
launches its own projection. The experimental `q8_verify_fanout` API quantizes
the full activation matrix once, then schedules projection tiles inside a CUDA
grid. A remaining partial tile uses the original narrow launcher. It supports
one to four Q8 projections with independent output widths and row strides.

`QW3_Q8_VERIFY_WIDE=1` enables this path only inside `Q8VerificationMath`;
it does not change ordinary prefill or serial decode. The default remains off.
`QW3_Q8_VERIFY_WIDE_TILE=32` selects the candidate with the strongest current
operator results. Supported experimental widths are 8 (the default), 16 and 32.
All preserve the decode-compatible Q8_1 quantizer, four-warps-per-row reduction
and FP32 accumulation order. No BF16 target weights or BF16 KV are introduced.
SwiGLU and residual-add projections still use the stable host-tiled reference.

The first layout interleaved activation tiles for adjacent thread blocks to
encourage weight-cache reuse. It regressed most tested shapes. It is retained
only as `QW3_Q8_VERIFY_WIDE_LAYOUT=interleaved` for ablation; normal layout
keeps weight rows in the original order. The first model run was deliberately
terminated with SIGTERM after the operator regressions were observed. Its
manifest is marked cancelled, not a failed model-correctness qualification.

## Operator evidence

`operator03` passes 66 original/extended linear cases and 84 multi-output
fanout cases for each 16/32-row configuration, plus the existing recurrent
checks. Wide outputs match serial matvec bit-for-bit. Guards cover differing
output widths/strides and padded input strides; 33/49/65-row cases exercise
partial tiles. CUDA memcheck reports zero errors for both widths.

The microbenchmark holds one projection group resident and measures warm
operator execution, including host launch overhead and a final synchronization.
It checks output equality on the timed input before measuring. Each shape uses
three alternating trials of twenty calls. These trials are within one process;
they are not the required independent end-to-end performance runs.

| Projection shape (output × input), fanout | 16 rows | 32 rows | 64 rows | 128 rows |
| --- | ---: | ---: | ---: | ---: |
| 5120 × 17408, one projection | 1.188× | 1.591× | 1.510× | 1.396× |
| 17408 × 5120, two projections | 1.217× | 1.314× | 1.254× | 1.237× |
| 5120 × 5120, four projections | 1.058× | 1.076× | 1.025× | 1.010× |

Values are median speedups of the 32-row candidate over the original host
8-row tiles. At a batch of sixteen the candidate falls back to 8-row GPU tiles.
The 16-row variant regresses the four-projection shape; it is not selected.
Raw identities, timings and failed experiments remain in
`q8_wide_fanout_validation_operator01/02/03.json` and
`q8_wide_fanout_operator_timing_summary01.json`.

## Scoped model qualification

The continuous probe alternates the reference and candidate for each CPU/GPU
sampling pair, reverses admission order and compares every emitted callback and
final response. Both normal completion and cancellation are checked. `model02`
passes **192 KVMem responses**: 64 from eight requests replenishing four slots,
and 128 from sixteen active requests. CPU and GPU sampling each compare the
reference directly with the 32-row candidate. Every paired group verifies the
same number of candidate-token rows. The host suite passes all 34 tests.

| Model diagnostic | Verification phase ratio | End-to-end ratio |
| --- | ---: | ---: |
| C4 replenishment, four CPU/GPU/cancel pairs | 1.041–1.159× | 0.967–1.014× |
| C16, four CPU/GPU/cancel pairs | 1.058–1.094× | 1.003–1.037× |

Ratios are reference time divided by candidate time. These are diagnostic
correctness probes with reversed request order, different live batch shapes,
cold request setup, short outputs and only one observation per pair. Initial
groups also overlapped host compilation; no other GPU job overlapped the model
runs. Thus the verification-phase improvements do not establish an end-to-end
throughput gain. No formal speedup, public-performance or default-enablement
gate has passed. The 32-row candidate also passes CUDA racecheck across all 150
linear/fanout cases: zero hazards, errors and warnings. This checks GPU kernels,
not the complete host-side concurrency lifecycle.

The full KVMem lifecycle, joint draft/prefill execution, serial-versus-CB
numerical quality, independent timing matrix and final integration remain open.

## Reproduction

Build the CUDA probe targets and use unique run IDs. Run GPU jobs sequentially:

```sh
python3 benchmark/dflash2/run_q8_wide_fanout_validation.py --run-id NEW_ID --cases operator-tile32 timing-tile32 memcheck-tile32 racecheck-tile32
python3 benchmark/dflash2/run_q8_wide_fanout_validation.py --run-id NEW_MODEL_ID --cases replenish-c4-tile32 model-c16-tile32
```

The runner records exact commands, environment overrides and source/binary/log
hashes and refuses to overwrite earlier evidence. The model probe sets the
wide-path flag to zero on each reference pass and one on its paired candidate
pass; this environment change occurs after all prior requests have completed.
