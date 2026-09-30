# Fixed-plan joint DFlash projections — 2026-09-25

This opt-in optimization passes its scoped correctness checks. The target remains Q8_0, target KV
remains E4M3 FP8, and only the official draft's BF16 projections change. The
persistent goal and all formal performance/lifecycle gates remain open.

## Arithmetic and implementation

The preceding joint-draft stage restored exact proposals by retaining a GEMM
per request. Direct `cublasGemmBatchedEx` is not an arithmetic-compatible
replacement: generated-input checks differ in 121 of 140 cases. Its source,
library/GPU identity and complete output remain in
`draft_cublas_batch_experiment01.json`.

The new search freezes an entire cuBLASLt plan selected for one request,
including its tile, split-K and reduction. It changes only the matrix extent
when combining requests. The generated-input experiment finds exact plans for
22 of 28 shapes, including all tested multi-token widths 3/7/8. Most width-one
shapes do not match, so single-row projections keep the verified private path.
This is measured behavior on the pinned GPU/library, not a cuBLAS-wide
guarantee that any heuristic choice preserves another algorithm's arithmetic.

`QW3_DFLASH_LT_DRAFT=1` enables the candidate in `dflash_linear_blocks`, after
the existing joint activation conversion. Plans are keyed by projection shape,
request width and total width. The heuristic sees one request; a subsequent
`cublasLtMatmulAlgoCheck` checks that same algorithm against the combined
matrix and the existing 32 MiB workspace. Supported plans execute one shared
GEMM. One request, width one, more than 32 requests, and unsupported plans use
the existing per-request GEMMs. Other target or prefill linear paths are
unchanged. Plans own their descriptors and are released with the backend.

## Evidence so far

`q8_draft_lt_validation_operator01.json` passes **2,576 bit-exact comparisons**
against private GEMMs: all 46 actual draft projection weights, every width 1–8,
and 1/2/3/4/8/16/32 requests. Both complete topology fixtures pass their 252
joint-request checks and bounded/snapshot continuations. CUDA memcheck passes
the 140-case representative-shape operator suite with zero errors.
The trace records 293 shared plans and one workspace-limited fallback
(1280 outputs, 5120 inputs, width eight, 32 requests). Actual single-row
projections and single requests use the private reference by design.

Warm operator timing (`timing02`) checks 28 representative weight/width/request
combinations with three alternating trials of twenty calls. The median ratios
are **2.966–9.120×** over per-request GEMMs, including activation conversion,
host dispatch and final synchronization. These are within-process operator
measurements, not independent end-to-end throughput runs. `operator01` timing
is retained but some lines interleave stdout/stderr; `timing02` disables plan
trace logging and supplies the complete timing table.

The real-model test alternates Lt off/on while retaining joint draft scheduling,
compares every output callback, and records exact proposal fingerprints.
`model01` passes **288 KVMem responses**: C4 replenishment (64), C16 (128),
repeated refresh (32) and direct private/joint comparison (64). Both CPU/GPU
sampling and cancellation pass. All **1,944 proposal comparisons** have identical
candidate/probability fingerprints and proposed token IDs. Repeated refresh
actually executes three refresh boundaries per non-cancelled request. The
34-test host suite passes. Formal speedup and default enablement remain open.
GPU target sampling records 2,856 calls with zero fallbacks. No default runtime
flag is enabled by this experiment.

## End-to-end limits and next bottlenecks

For C4 replenishment, draft phase ratios are 1.384–1.515× but end-to-end ratios
are 0.973–1.006×. For C16, draft work falls from about 0.66 s to 0.36 s in the
normal GPU-sampled group, while total request time is about 40 s. Summed
per-request accounting identifies about 16.5 s of prefix computation and 2.4 s
of post-prefix selection/replay; ragged target verification takes about 5.4 s.
Substantial remaining time is outside these named phases. It must be measured,
not attributed to an assumed bottleneck. Reversed admission order, different
live batch shapes, cold setup and within-process repeats make these diagnostic
timings unsuitable for the formal independent throughput gate.

| Diagnostic case | Draft phase ratio | End-to-end ratio |
| --- | ---: | ---: |
| C4 replenishment | 1.384–1.515× | 0.973–1.006× |
| C16 | 1.841–1.949× | 0.995–1.052× |
| C4 repeated refresh | 1.506–1.604× | 1.004–1.035× |
| Private draft versus optimized joint draft, C4 | Not comparable in this trace | 0.988–1.007× |

Ratios divide reference time by candidate time; each range contains four
CPU/GPU/normal/cancel pairs. These are not comparisons against none or MTP.

The next work is full scheduler/controller accounting and matched KVMem
baselines. Current frontend code routes guided/refresh none/MTP requests through
the serial transactional controller, while the opt-in DFlash cold controller
can share verification. Thus the existing QC-disabled none/MTP CB probes are
not equivalent controls for the complete DFlash policy. Same-policy baseline
coverage, joint prefill, serial/CB numerical quality, cached/archive/repository
lifecycle, the frozen required matrix and public comparison remain open.

## Reproduction

Use distinct IDs and run GPU jobs sequentially:

```sh
python3 benchmark/dflash2/run_q8_draft_lt_validation.py --run-id NEW_OPERATOR --cases linear fixture-fp32 fixture-bf16 timing memcheck
python3 benchmark/dflash2/run_q8_draft_lt_validation.py --run-id NEW_MODEL --cases replenish-c4 kvmem-c16 refresh-c4 private-parity-c4
```

The previous joint-draft manifest remains frozen for its original sources.
Current per-run manifests identify this optimization's sources and binaries.
