# Method and efficiency audit

## Native implementation checks (completed first)

InfiniGen's untouched speedup implementation runs OPT/FlexGen, with the author's
calibration, threshold selector and prefetch loop. It is not a native Qwen
backend. Its generic overlap loop does not execute the same selective prefetch
path, so enabling that flag merely to improve throughput would be invalid.
CPU/full-GPU KV controls and batch-one/batch-four runs were measured separately.
OPT's 1920+128-token configuration stays within its positional capacity.

InfLLM's untouched patch_hf implementation runs Mistral. Author-supported fused
attention, pinning, stream and thread choices were checked. The original
shared-memory fallback selects a smaller Triton tile on this Blackwell GPU.
Pinning helped; disabling the separate global stream was slightly faster in
calibration. Unbound 120K repetitions varied; GPU-local CPU affinity produced
three stable repetitions. Both sets remain in the auxiliary export.

## Qwen integration

All comparison arms now use the same original BF16 safetensors. The old Q8
adapter is excluded from this campaign because it calls qw3 projection kernels.
External model projections here are ordinary PyTorch BF16 operations, with
HF/FLA hybrid recurrence and the optional causal-conv1d CUDA extension enabled.
The process report verifies that libaligned_q8.so was not loaded.

InfLLM reuses the original ContextManager, block representatives, selection,
LRU cache and two-part fused attention. Necessary boundary changes retain Qwen
GQA, head RMSNorm, partial RoPE, output gate and 48 recurrent layers. Head dim
256 requires extending the original kernel shape guard and reducing pipeline
stages; the math is checked against the author's Torch attention oracle.
FP8 history is an explicit extension. Persistent recent working views contain
exact FP8-rounded values in FP16; keeping those views eliminates repeated
lossless conversion without changing attention output or selected block order.

InfiniGen's Qwen port retains cross-layer forecast, SVD calibration, the 20%
partial subspace, alpha=4 and the author's threshold/mean-count/top-k rule. GQA
queries map to four shared KV heads. Qwen's head normalization and RoPE mean a
full Q projection must precede subspace projection; unlike the original OPT
path this cannot be replaced by a fixed folded partial projection. That real
Qwen adaptation cost remains included. Counts accumulate in FP32 to prevent
FP16 overflow on >65K histories; a long-count regression verifies the rule.

A material model-specific limitation is the partial-index footprint. Qwen has
24 query heads but only four KV heads. The existing per-query-head FP16 index
stores 51 retained dimensions, or 24×51×2=2448 bytes/token/full-attention-layer.
Full shared-head FP8 K+V stores 2×4×256=2048 bytes. Thus this port's index alone
is 1.195× the full FP8 KV size (8.965 versus 7.5 GiB at 240K across 16 layers).
GQA row deduplication reduces transfers but cannot merge different learned
per-query-head subspaces. Quantizing/restructuring that index would require a
separate numerical/method variant. The current results must not be presented
as evidence that InfiniGen generally reduces Qwen FP8 GPU memory; see
`infinigen_index_footprint.json` for the exact calculation.

Two actual gather paths were timed at 32K on held-out calibration history:
original per-query-head gather (7.65/7.69 token/s at 2/8 CPU threads) and exact
GQA row deduplication (8.87/9.23 token/s). The latter is selected. It gathers and
transfers each identical (token, KV-head) row once, then restores the identical
per-query-head order on GPU. Selection scores, token sets, precision and
refresh frequency are unchanged. Byte-level gather checks and real-model
full-selection tests validate the change.

## Execution controls and validation

- GPU-local CPU affinity for every arm; exclusive GPU, no simultaneous OSWorld.
- Fused causal convolution and FLA recurrence are enabled for all HF arms.
- Optional graphs capture only MLP/recurrent computation, never retrieval.
  They verify exact outputs AND recurrent state against uncaptured execution.
  Graphs improved the 32K dense calibration from 19.64 to 19.92 token/s.
- InfLLM 2 versus 8 threads produced 15.55 versus 15.58 token/s; choose 2.
- InfiniGen 8 threads was faster than 2; choose 8. Calibration is outside timing.
- InfiniGen full selection: eight real-model decode steps have zero logit
  error against GPU dense FP8, and CPU historical KV bytes match exactly.
- Full CPU offload control: eight real-model decode steps and historical KV
  bytes match GPU dense FP8 exactly.
- Native KVMem successfully selects and executes full query replay at 240K.
  Its raw plain decode counter includes initial maintenance, so throughput
  uses the first-model-token-to-completion interval instead.

The formal native KVMem run also exposes a rollback limitation: repeating the
identical 240K prompt removes the previous 127-token generated suffix. Its P
checkpoint is marked non-resumable after selection, while current M lies after
the new LCP. Two requests fall back with no_reusable_checkpoint and cold-prefill
for 109.575/108.352 seconds. Preserve these times; no production-code repair is
folded into the frozen throughput experiment. This is a repeated-prefix rollback
case, not evidence of its prevalence in append-only Claude Code trajectories.
See rollback_checkpoint_observation.json. Future end-to-end claims require a
separate check of this case and the actual agent request pattern.

These are bounded engineering checks, not a proof of globally optimal tuning.
The native engine's FP32 residual and HF's BF16 residual differ, as do some
operator and positional FP8 rounding boundaries. Report same-model system
throughput with those limits; do not attribute the entire ratio solely to the
KV selection algorithm or infer agent accuracy from fixed-prefix decoding.

## Exact InfLLM CPU block-map optimization

The block=128, 128K-budget original path selects 991 blocks per layer and its
get_global_hidden_and_mask method nests linear list-membership searches.
The isolated fast_block_map.py wrapper substitutes sets for membership tests
and an ascending free-slot iterator. It keeps the selected IDs, sorted order,
buffer slot choice and load/copy order unchanged; no GPU math or retrieval
frequency is reduced. The original method is recovered with inspect and exact
source anchors, and the generated method source is saved for review.

CPU validation exercises the actual original and optimized functions with
identical buffers at top-k 1, 4, 16, 64, 128 and 991; all outputs, slot maps
and copy calls match. A separate full-model 240K-input/32-output check compares
all generated IDs before accepting the optimized result. See
block_map_validation.json and block_map_model_validation.json (written only
after the real-model comparison passes). Both original and optimized runs are
retained. The upstream checkout remains unmodified. This changes bookkeeping
only; the separately reported block=2048 alternative changes retrieval
granularity and must not be described as numerically equivalent.
