# Q8_0 / FP8-KV scope and qualification

The user's explicit precision target is Qwen3.8-27B **Q8_0 weights + FP8 KV**,
with KVMem as the performance deliverable and ordinary QW3 as a regression
control. The DFlash 2 goal remains active and incomplete.

The [fixed-plan projection candidate](dflash2_draft_lt_20260925.md) passes
2,576 exact operator cases, memcheck, 288 KVMem responses and 1,944 exact
proposal comparisons. Its operator timing does not qualify whole-model
throughput or public performance, and the candidate remains opt-in.

The [joint draft stage](dflash2_joint_draft_20260925.md) passes its scoped matrix.
Replenishment exposed draft GEMM
arithmetic that depended on batch size; fixed per-request GEMM shapes pass
140 exact linear cases and 64 private/joint replenishment responses, including
495 exact proposal fingerprints. The complete scope passes 352 responses
(320 KVMem), C16, cancellation, guided queries and refresh, both topology
fixtures, CUDA operator memcheck/racecheck and 34 host tests. This carries no
formal performance or complete lifecycle qualification.

The current [wide Q8 verifier experiment](dflash2_wide_q8_verifier_20260925.md)
has passing operator parity/memcheck evidence and 192 KVMem model responses
versus the reference. CUDA racecheck reports zero hazards. It is opt-in and does not change
the formal performance status below.

Subsequent [KVMem lifecycle qualification](dflash2_kvmem_lifecycle_20260924.md)
extends these short tests to 4K/32K histories, query-policy equality and paired
compaction state. It fixes a separate multirow attention-merge discrepancy and
adds complete draft snapshot cloning/offload. Historical findings below retain
their original scope and per-run binary identities.

The [archive stage](dflash2_kvmem_archive_20260925.md) subsequently validates
paired target/draft ladder persistence, 4K/8K retained-history reload and exact
resumed-build bytes. It adds no BF16 target/KV support or public performance claim.

The subsequent [GPU sampling stage](dflash2_gpu_sampling_20260925.md) passes its
operator and initial CPU/GPU model checks, and uncovers FP8 cache drift during
longer repeated timing. Native FP8 now reconstructs every moved K block from raw
authority. Its recorded short regression and diagnostic timing pass; historical lifecycle/archive
results below use the previous remapping policy and do not qualify the entire
updated configuration by themselves.

The [continuous-sampling stage](dflash2_continuous_sampling_20260925.md) passes
320 dense none/MTP responses at C4/C16 after fixing external MTP page addressing.
The subsequent [shared KVMem stage](dflash2_shared_kvmem_pool_20260925.md) resolves
its immutable-K allocator obstacle with request quotas over the shared arenas.
Scoped KVMem none/MTP C4/C16 probes pass. It also fixes silently delegated joint
drafts and stale shared checkpoints after a batch shrinks to private verification.
The newer [DFlash controller stage](dflash2_continuous_controller_20260925.md)
connects the complete cold request control flow to joint target verification and
covers C4/C16 query-conditioned replay, completed private queries and repeated
refresh. Replenishment exposed a CPU repeat difference under changing batches.
A controlled correction retains ragged verification and decode-compatible Q8
tiles; its 64 responses pass and a disabled-tiling control fails. The final default
matrix passes 448 responses (416 KVMem), with 1,611 actual joint DFlash target
batches and zero GPU-sampling fallbacks. Single-request none/MTP/DFlash
regressions also pass all 24 CPU/GPU pairs (48 responses);
all failed and passing records are retained. Proposals remain individual; shared
prefix/archive/session/compaction and bounded-repository CB remain open. These
scoped results and CB's FP32 intermediate activations do not qualify the complete
KVMem or serial-Q8 performance configuration.

## Current arithmetic alignment — 2026-09-24

The previously failing short KVMem identity and Chinese dense cases now pass.
The corrected verifier preserves serial Q8 decode's convolution accumulation,
four-warp matrix reduction, Q8_1 activation scales, and fused residual rounding.
Tiny-batch Q8 SwiGLU is fused; the batch-8 verifier head uses the MMVQ fanout
route instead of silently switching to general GEMM's FP32 activation scales.
Ordinary single-token decode keeps its existing arithmetic.

The real-model target-only probe now has an optional `--require-exact` gate.
All widths 1–8 at four prefix positions pass exact equality of all 64 layer
residuals and full vocabulary logits. Math, code and Chinese each pass 64-token
greedy integration, all seven rejection boundaries and paired snapshots pass.
The normal and bounded KVMem production probes each pass 32 short checks.
Ordinary QW3 passes 44 production checks, and the 256-token KVMem timing
comparison passes identical greedy output on both routes for all requests.
These are local regression workloads, not whole-model quality or public
performance qualification.

The new `qw3-q8-verify-kernels` CTest checks 24 Q8 linear configurations (including
guarded strides and internal BF16 residual rounding), 48 temporal recurrent
configurations with/without checkpoints, and 24 independent plus 24 ragged
recurrent configurations. It exposed a shared-memory read/write race in the
independent DeltaNet kernel: scratch was overwritten before all warps consumed
the preceding reduction. The same unsafe reuse existed in four fallback
DeltaNet variants; all now synchronize after broadcasting the reduction.
Racecheck reported 1,404 hazards before the fix and zero afterward. Memcheck
also reports zero errors. The original `qw3` and ported recurrent selections
both pass the numerical operator suite; all six selected CUDA CTests pass.

Authoritative runs: `benchmark/dflash2/q8_kernel_alignment_01.json` (initial
real-model evidence), `q8_alignment_validation_02.json` (post-barrier validation),
and their hashed logs. Earlier test-harness readback/guard failures are retained
in logs and do not count as passing runs. The public performance gate, long
history, archive/compaction, GPU sampling and DFlash continuous batching remain
open. See the implementation status document for subsequent timing results.

Target provenance is now verified: local SHA-256
`a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`
and byte size match the
[upstream Q8_0 file](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf).
Use `target_q8_manifest_v2.json` and `public_execution_protocol_v6.json`;
earlier manifests retain their historical identities.

## Removed at the user's request

- Unquantized HF BF16 target loading; the original compressed-tensors gate is
  restored. The raw safetensors loader used by the DFlash draft remains.
- BF16 target/MTP KV allocation, local/global paged-pool dtype support and
  BF16 KV append/FlashInfer attention dispatch.
- `--kv-dtype bf16`, the dedicated BF16 KV test file and both BF16-KV CTest
  entries; FP16/FP8 attention tests and interfaces are restored.
- BF16 target transform tests added specifically for that now-removed loader.

The official BF16 DFlash draft, its numerical oracle, and the framework's
pre-existing BF16 internals remain. In particular, the Q8 target already uses
BF16 residual storage by default; that does not mean its **weights** are BF16.
No original-worktree files or model weights were deleted. Historical logs,
sources and binary hashes are retained as inactive benchmark artifacts.

## Validation after removal

- Release CUDA/FlashInfer and host builds succeed.
- All 32 host CTests pass.
- Five selected CUDA attention CTests pass: FP16/FP8 decode, prefill, queued
  metadata lifetime, plan cache and the FP8 gate oracle.
- The actual official 81-tensor draft checkpoint still loads.
- CLI rejects BF16 KV; the native loader rejects the unquantized BF16 target.

See `benchmark/dflash2/q8_only_host_ctest.log`,
`q8_only_attention_ctest_01.log`, and `q8_only_scope_validation_02.json`.
Per-run manifests identify exact tested binaries; historical BF16 results are
not relabeled as Q8 successes.

## Historical serial/batch discrepancy (short cases resolved above)

The KVMem all-fit math prompt diverges at zero-based output callback 21:
serial produces “Let me work this out”, DFlash produces “Let me work through
this”. The failure is recorded before and after residual writeback correction
in `q8_fp8_kvmem_parity_02.log` and `q8_only_kvmem_01.log`.

The new target-only layer diagnostic uses the same prefix snapshot and teacher
tokens, captures all 64 layer residuals and compares batches 1/2/4/7/8 against
serial execution, without any draft calls. Even batch 1 differs before the fix,
beginning in recurrent layer 0. This is stronger localization than comparing
only final generated text; it does not yet prove the complete cause of the
KVMem failure.

One concrete discrepancy is corrected: serial Q8 decode adds the FP32 output
projection to its BF16 residual and rounds once. The tiny verifier previously
rounded the projection to BF16 first, then added/rounded again. Verification
now uses the existing fused residual writeback for both attention/recurrent
output and FFN down projection. Ordinary prompt prefill remains unchanged.
Real-model output parity still fails, so this correction is not presented as
the complete solution or as a validated performance improvement.

The rebuilt dense real-model probe passes both math and code prompts at 64
tokens, all seven rollback boundaries and paired snapshots. Chinese diverges
at output index 29 (expected token 104257, actual 98183). Replaying the same ten
target verification blocks with no draft calls exactly reproduces the verifier
logits. This isolates a target execution numerical difference for that case;
it does not close KVMem or whole-model quality gates. See
`q8_only_q8_target_02.log` and the per-run binary hash.

Further numerical differences to investigate include recurrent convolution
accumulation order (serial oldest-first versus batch newest-first), the Q8
matrix kernel's batch-dependent reductions/scales and attention reduction
paths. Do not replace the baseline with a slower or different-precision target
to make the existing test pass, or report a speedup from a failing output gate.

The public/reference workload, long-history KVMem correctness/quality, GPU
sampling and DFlash continuous batching remain outstanding. The earlier SGLang
BF16 reference requires new precision/quality qualification before comparison
with this Q8_0/FP8 deployment.
