# DFlash 2 implementation status — 2026-09-24

The persistent goal is **active and incomplete**. Implementation worktree:
`/home/chaidi/qw3-dflash2`, branch `codex/dflash2-qwen38`. The user's original
worktree remains separate. No public performance gate has passed.

Current milestones and acceptance criteria:
[Q8_0 + FP8 KVMem goal](dflash2_q8_fp8_goal_20260924.md).

Current optimization: [fixed-plan joint projections](dflash2_draft_lt_20260925.md)
passes 2,576 exact real-weight linear comparisons and operator memcheck. Warm
projection timing improves; full-model parity passes 288 KVMem responses and
1,944 exact proposal comparisons. End-to-end qualification remains open. It is
opt-in; prior stage manifests retain their original scope. Next steps include
complete scheduler accounting and same-policy none/MTP controller coverage.

Current work: [joint DFlash proposals](dflash2_joint_draft_20260925.md) batch the
draft operations and selector while preserving private caches and RNG.
Replenishment exposed batch-dependent draft GEMM arithmetic. Fixed per-request
GEMM shapes with joint activation conversion pass 140 exact linear cases and
64 private/joint replenishment responses (495 exact proposal fingerprints).
The completed scope passes 352 model responses (320 KVMem), C16, cancellation,
guided queries and repeated refresh, both topology fixtures, CUDA operator
memcheck/racecheck and 34 host tests. The feature is opt-in, with no performance
qualification; fixed-shape projection GEMMs still execute per request.

Current experiment: [wide Q8 verification](dflash2_wide_q8_verifier_20260925.md)
quantizes each projection batch once and keeps decode-compatible reduction.
The 32-row candidate passes operator parity/memcheck and 192 KVMem model
responses versus the reference. CUDA racecheck reports zero hazards. It remains opt-in and
has no formal speedup claim.

Latest work: [DFlash continuous KVMem controller](dflash2_continuous_controller_20260925.md)
adds opt-in cold-request DFlash admission with shared target verification, draft
feature capture and per-request acceptance/checkpoint/logit transport. Its
C4/C16 KVMem and dense checks pass, along with private-query completion and
repeated refresh. Queued replenishment exposed batch-dependent Q8 arithmetic.
A controlled tiled-Q8 correction passes 64 responses; disabling it reproduces
the CPU repeat failure. The final default matrix passes 448 responses (416
KVMem), with 1,611 actual joint DFlash target batches and zero GPU-sampling
fallbacks. Single-request none/MTP/DFlash regressions pass all 24 CPU/GPU pairs
(48 responses). Drafts still execute individually. Prefix,
archive/session/compaction and bounded-repository CB remain open, as do CB's
existing FP32-versus-serial-Q8 numerical quality and formal performance gates.
The earlier [shared KVMem pools](dflash2_shared_kvmem_pool_20260925.md) stage
corrected MTP joint-draft metadata and shared-to-private checkpoint transitions.

The preceding [GPU sampling and FP8 remapping](dflash2_gpu_sampling_20260925.md)
moves native single-request target sampling/acceptance onto CUDA. Timing exposed
FP8 delta-RoPE drift across repeated cached requests; moved FP8 K now rebuilds
from immutable raw authority. Its recorded regression and diagnostic timing pass;
earlier archive/long-history evidence retains the old remapping policy and needs
confirmation before whole-goal acceptance.

The prior [archive stage](dflash2_kvmem_archive_20260925.md) adds bounded
draft ladder persistence, exact paired reload and resumed-build equality.
Its per-run manifests distinguish the earlier CLI matrix from the final
payload-overflow guard, zero-ladder fix and follow-up validation.
The earlier [lifecycle qualification](dflash2_kvmem_lifecycle_20260924.md) covers
4K/32K bounded history, compaction state and explicit query/refresh policy
equality. Remaining whole-goal gates below remain open.

## Current user scope

Primary deployment: **Qwen3.8-27B Q8_0 target weights + FP8 KV in KVMem**.
Ordinary QW3 needs correctness/regression qualification only. The user explicitly
requested removal of the new BF16 target-weight and BF16-KV support. The ordinary
HF BF16 target loader extension, BF16 KV allocation/append/attention adapters,
CLI option and dedicated test entries have been removed. Official DFlash draft
BF16 weights and pre-existing framework arithmetic remain necessary, including
the Q8 path's internal BF16 residuals. Historical BF16 runs are archived evidence,
not supported configurations or substitutes for the deployment baseline.

## Implemented

- Official 81-tensor DFlash 2 draft loader, with standalone/sharded safetensors
  validation and separate target/draft transforms. Draft revision:
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- Five-layer draft with noncausal sliding GQA attention, dynamic convolution,
  top-16 selector and actual proposal probabilities. Draft linear weights are
  BF16, residual/attention/cache FP32. Target taps are layers 5/19/33/47/61.
- Bounded draft window, uint64 logical positions, raw-K RoPE rebasing, paired
  target/draft snapshots and rollback. Target-only restore clears stale draft
  context. Rejected rows never enter committed draft history.
- Native greedy and distribution-preserving speculative acceptance/residual
  sampling, both target filter orders, penalties, EOS/thinking/cancellation.
  `--dflash-model` is mutually exclusive with built-in MTP.
- KVMem normal/bounded query replay, warm prefix restore and tail rewrite.
  Earlier tests exposed and fixed wrong replay positions, stale checkpoint
  admission and released CPU-index GPU-buffer capacity.
- FP8 contiguous small-batch attention gate fix, retaining one sigmoid in the
  gated wrapper. A 24-case independent analytic oracle fails before the fix,
  passes afterward (maximum error 1.19209e-7), and passes memcheck.
- Plain KVMem timing version 2 separates post-prefill selection/replay from
  decode, retaining the old combined duration explicitly in logs.
- Q8 tiny-batch verification now uses the serial residual writeback operation.
  Previously it rounded a projection to BF16 before adding the residual, while
  serial decode rounded once after addition.
- Q8 verifier arithmetic now matches serial convolution order, MMVQ reduction
  width and activation scales. Added batched Q8 fused SwiGLU and a decode-compatible
  batch-8 output head. Shared-memory broadcast barriers fix a demonstrated race
  in four DeltaNet fallback paths.
- Multirow attention merge preserves the serial reduction kernel even when
  verifier rows outnumber KV tiles. Immutable draft snapshots survive cloning
  and CPU offload/restore. Target-only advanced KVMem requests use the shared
  transaction controller without MTP, and speculative cycles respect refresh
  boundaries.
- Optional draft archive ladders preserve the bounded raw-K/V window and RoPE
  frame, with configuration/weight identity and validated restoration. Explicit
  target-only archive mode no longer becomes MTP implicitly. The same target KV
  archive is usable with and without DFlash; legacy archives log a cold draft.
- CUDA target sampling, penalties, both filtering orders, sparse/point-mass
  acceptance and residual/bonus draws. Sampling copies a 16-byte decision to
  host; ties/numerical boundaries explicitly fall back without consuming RNG.
- FP8 position changes rebuild K from immutable raw authority; stable resident
  positions skip reconstruction. This fixes a reproduced CPU-only repeated-cache
  drift, and does not add BF16 target/KV support.

## Evidence and remaining gates

- Host regression suite: all 34 tests pass, including cooperative controller
  stack ownership and teardown. The controller test also passes ASAN/UBSAN;
  TSAN cannot initialize on this host.
- Five selected FP16/FP8 CUDA attention CTests and the new Q8 verifier operator
  CTest pass on the rebuilt implementation; the actual
  81-tensor draft loads, and BF16 target/KV requests are rejected as intended.
- Draft operators: conv, attention, top-k, selector and feature packing pass;
  memcheck zero errors and racecheck zero hazards in their recorded stages.
  Full official-topology FP32 and BF16-weight draft fixtures pass explicit
  precision bounds; these validate the draft, not native BF16 target support.
- Target sampling: 1,444 analytical/random checks and 60 pinned upstream
  distribution cases pass, including real-vocabulary nucleus ties.
- GPU sampling: 1,215 operator/upstream cases and 6,000 independent marginal
  trials pass. Full operator memcheck and bounded racecheck report zero errors.
  Both initial and corrected FP8 real-model matrices pass 24 same-route pairs.
  The corrected default also passes 24 cold/warm CPU cache requests, 32 normal
  and 32 bounded KVMem checks, and 12 sampled timing pairs. Earlier failed
  mixed-cache and repeated warm timing runs remain recorded.
- Current Q8_0/FP8 real-model target probe: math/code/Chinese each match for 64
  tokens; seven rollback boundaries and paired snapshots pass. The former
  Chinese token-29 and KVMem callback-21 failures are resolved on these cases.
- Target-only strict regression: widths 1–8, four positions, all 64 layer
  residuals and full vocabulary logits are bit-exact against serial decode.
- Normal and bounded KVMem production routes each pass 32 short checks. Full
  long-history and advanced KVMem path qualification remains open.
- Ordinary QW3 production route passes 44 checks. The 256-token KVMem timing
  probe also passes identical greedy output on both routes for every request.
- New verifier operators pass 24 linear and 96 recurrent configurations,
  including state checkpoints, independent requests and ragged batches. Both
  recurrent kernel selections pass. Racecheck: zero hazards after a confirmed
  1,404-hazard failure before the broadcast barrier; memcheck: zero errors.
- DFlash continuous batching is **not implemented**; requests are serialized.
- Recorded bounded 4K/32K history, guided/semantic replay and compaction state
  oracles pass. Retained-history named-cache fork/append/version/eviction also
  passes. Recorded 4K/8K retained-history archive reload and resumed-build byte
  equality pass. Archive HTTP/large bounded APIs, production compaction quality,
  exhaustion/soak and FP8 whole-model quality qualification remain open.
- Public workload inputs are frozen: five datasets, 128 measured + 32 warmup
  rows/task; all 800 rendered xhigh inputs / 107,581 token IDs match.
- Pinned same-GPU SGLang reference smoke runs passed with CUDA graphs, but used
  BF16 target weights/KV. That is not a matched-precision Q8_0/FP8 baseline.
  Reference precision/quality and actual admission capacity require qualification.
- Formal repeated KVMem performance/public comparisons have **not run**.
- Q8 GGUF bytes and SHA-256 match the upstream Unsloth artifact, pinned at
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`. Current target manifest and comparison
  protocol: `target_q8_manifest_v2.json`, `public_execution_protocol_v7.json`.

## Corrected deployment baseline

Current canonical-FP8 greedy rerun: end-to-end **44.933 → 85.617 tok/s**, decode
**45.59 → 88.70 tok/s**, with matching outputs. This is the same short diagnostic
configuration described below, with the corrected reconstruction policy; it is
not formal or public acceptance. The separate 4K sampled CPU/GPU optimization
diagnostic gives DFlash **78.965 → 112.903 tok/s** and MTP **85.880 → 113.892**.
That sampled observation does not establish DFlash superiority over MTP.
Raw records: `q8_gpu_sampling_summary_final03.json`.

Historical lifecycle-stage rerun (before the FP8 remapping correction): warm
end-to-end median **44.235 → 87.668 tok/s**
(1.982×), decode **44.92 → 90.64 tok/s**. It uses the same diagnostic configuration
and limitations below. All outputs match. Raw samples:
`benchmark/dflash2/q8_lifecycle_timing_summary_01.json`. This is not formal
performance acceptance or a public implementation comparison.

After arithmetic alignment, a same-configuration 256-token greedy KVMem
diagnostic passes output equality on every request. Hardware is unchanged;
history/query lengths are 833/848, working set 256, generation reserve 256,
block size 32 and concurrency 1. Three warm repeats are within one process
per route (plain first, DFlash second), not independent alternated benchmark
runs. Exact callback count divided by request wall time gives:

| Scenario | Plain end-to-end tok/s | DFlash end-to-end tok/s |
| --- | ---: | ---: |
| Short all-fit request, one run | 45.37 | 156.97 |
| First sparse query, one run | 37.50 | 64.15 |
| Warm sparse query, median of three | 44.22 | 87.38 |

Warm decode medians are 44.87 and 90.36 tok/s, respectively. Warm end-to-end
ratio is 1.976×; ranges are 44.13–44.52 (plain) and 87.02–88.65 (DFlash).
These are diagnostic results with the same outputs, not a public benchmark or
the goal's required performance matrix. Raw phases, TTFT, wall times and hashes
are in `benchmark/dflash2/q8_alignment_timing_summary_02.json` and
`q8_alignment_validation_02.json`. MTP and qualified public baselines are still
required before formal performance acceptance.

The earlier baseline audit below is retained to explain the original discrepancy.

On the RTX PRO 6000 Blackwell, no MTP/DFlash, Q8_0 weights + FP8 KV, 256 output
tokens and three warm repetitions of the fixed short sparse query: native decode
median **45.61 tok/s**, independent token-callback rate **45.44 tok/s**, request
end-to-end **44.95 tok/s**. This is a local diagnostic, not the public benchmark.
The previous ~21 tok/s number used BF16 weights and a plain decode timer that
included selection/replay. Dividing the previous DFlash ~85 by ~21 is invalid.
See `docs/dflash2_fp8_kvmem_20260924.md` and
`benchmark/dflash2/baseline_timing_audit_01.json` for the raw definitions/data.

## Reproduce the current scope

Use the existing Release CUDA 13 / SM120a / FlashInfer build from this worktree.
The following are diagnostics with strict output checks; an open gate may fail.

```bash
cmake -S . -B build-dflash-flashinfer
cmake --build build-dflash-flashinfer --target qw3 qw3-dflash-backend-probe qw3-dflash-target-probe qw3-target-batch-parity-probe -j 4
QW3_KV_DTYPE=fp8 build-dflash-flashinfer/qw3-dflash-backend-probe models/Qwen3.8-27B-Q8_0.gguf models/Qwen3.8-27B-DFlash2-BF16 kvmem
QW3_KV_DTYPE=fp8 build-dflash-flashinfer/qw3-dflash-backend-probe models/Qwen3.8-27B-Q8_0.gguf models/Qwen3.8-27B-DFlash2-BF16 kvmem-bounded
QW3_KV_DTYPE=fp8 build-dflash-flashinfer/qw3-dflash-target-probe models/Qwen3.8-27B-Q8_0.gguf models/Qwen3.8-27B-DFlash2-BF16 64
QW3_KV_DTYPE=fp8 build-dflash-flashinfer/qw3-target-batch-parity-probe models/Qwen3.8-27B-Q8_0.gguf 19 --require-exact
```

The layer diagnostic uses identical target prefix snapshots, teacher tokens and
all 64 layer residuals, with no draft calls. `--require-exact` fails on any residual,
logit or argmax difference; without it, process success only means the diagnostic
completed.
The regression probe's `--plain-timing` mode loads no draft and runs the longer
256-token baseline with three warm repetitions; `--timing` additionally runs
DFlash and requires matching greedy text and token counts for every request.
Scope removal and current parity evidence are detailed in
`docs/dflash2_q8_fp8_qualification_20260924.md`.

Historical sources/logs are retained under `benchmark/dflash2`; per-run SHA-256
hashes identify the actual binaries. Removed BF16-stage sources are archived in
`retired_bf16_sources_20260924`, outside active source/test targets. Earlier build
binaries may remain as historical artifacts; only rebuilt current targets count
as evidence of the current source. No commit, PR or deployment has been made.
