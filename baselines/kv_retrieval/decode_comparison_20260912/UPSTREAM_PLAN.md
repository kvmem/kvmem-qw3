# Native upstream decode comparison

User steering, 2026-09-12: reproduce the authors' open-source implementations,
without using qw3 as their backend. Earlier Qwen adapter tuning in this directory
is diagnostic only and is not the primary comparison.

## Implementations

- InfiniGen: pristine `../upstream/InfiniGen`, commit
  `3754f52d131c30ab7a310e3c4cf356e8e3326ac3`; its OPT/FlexGen speedup pipeline,
  original FP16 weights/KV, original selector, skewing, partial weights and CPU
  KV movement. Use OPT-6.7B, PG19 input supplied by the authors, 1,920 input and
  128 output tokens (within OPT's 2,048-token position limit). Compare against
  the authors' original FlexGen CPU KV baseline and GPU-resident KV control.
  Batch one is the agent-relevant result; additionally test a larger batch to
  distinguish per-agent latency from aggregate serving throughput.
- InfLLM: pristine `../upstream/InfLLM`, commit
  `12b70798f56e56ebb23c53c7018091a3f540a028`; original `patch_hf` with official
  Mistral-7B-Instruct-v0.2 and BF16 weights/KV. Official block/local/retrieval
  budgets, original Triton attention, author-supported async/pinning options.
  Compare with the same Mistral model's GPU-resident HF SDPA decode. Test
  history lengths separately; do not interpret beyond-window dense decode as
  an accuracy result.
- KVMem / ordinary qw3: remain on the frozen Qwen3.8-27B Q8_0 program. These
  are separately labeled deployment references, not same-model speedup ratios
  against OPT/Mistral. Report MTP=0 and MTP=4 separately. Both use real current
  query spans and full semantic replay; no private generated query is needed
  in this raw text decode workload. For MTP=0 the configuration must disable
  guided generation because the native server requires MTP for that option.

## Reproducibility and timing

Use an isolated environment and GPU exclusively. Preserve pristine upstream
checkouts; put any required dependency/hardware compatibility changes in this
directory and record exact diffs. No algorithm reimplementation or Qwen
projection reuse. Original FP16/BF16 precision takes precedence over the prior
permission to use FP8. Download without any HTTP/HTTPS/ALL proxy, using
hf-mirror, pinned revisions and exact downloaded model snapshots.

Separate prefill from decode, synchronize the GPU at measurement boundaries,
include attention selection and transfers in decode, and report generated
tokens / measured decode seconds. Exclude model loading, one-time calibration
and kernel warmup, but disclose them. No projected timing, dummy weights or
skipped retrieval. Keep per-run raw timing, source/configuration metadata and
repeat measurements; report batch size and output count explicitly.

Calibration: native InfiniGen was checked at 1/2/4/8 CPU threads; original
FlexGen additionally checked its overlap option. Its fastest tested controls
used the author's overlap=false setting. InfLLM was checked at 8K and 32K,
including native Triton, pinning, async/sync stream, and thread counts. The
32K calibration exercises histories larger than the resident history block
cache. Use a different recorded history for formal Mistral/Qwen measurements.
This is bounded tuning on the current machine, not a claim of a global optimum.

OSWorld is paused. Preserve completed arms and the interrupted 002/KVMem
attempt. Once measurements and report are complete, restart the frozen suite
and verify it reuses completed results and resumes at the interrupted arm.
