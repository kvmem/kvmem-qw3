# External Qwen Q8 MTP throughput

Medians of three repetitions, token/s. One RTX PRO 6000 Blackwell, exclusive
GPU, CPU affinity 0-15,32-47, two Torch CPU threads. Exact Qwen3.8-27B Q8_0
GGUF; FP8 historical KV; greedy generation, 128 outputs (127 timed outputs).
The complete decode interval includes draft, every speculative retrieval,
transfer, rejected verification, sampling, state restoration and MTP priming.
Model loading, prompt prefill, calibration and graph setup are excluded.

| Input | InfLLM MTP off | InfLLM MTP=4 | ratio | InfiniGen MTP off | InfiniGen MTP=4 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 21.39 | 22.41 | 1.05x | 17.38 | 8.34 | 0.48x |
| 32K | 20.39 | 20.47 | 1.00x | 11.17 | 8.39 | 0.75x |
| 120K | 19.49 | 17.59 | 0.90x | 8.70 | 6.72 | 0.77x |
| 240K | 18.57 | 18.44 | 0.99x | 7.99 | 6.64 | 0.83x |

Previously measured native deployment references (token/s):

| Input | Native qw3 Q8 MTP=4 | Native KVMem Q8 MTP=4 |
|---|---:|---:|
| 8K | 115.04 | 111.66 |
| 32K | 106.74 | 93.18 |
| 120K | 94.00 | 89.77 |
| 240K | 73.32 | 72.59 |

## Correctness and method provenance

InfLLM uses the pinned author's ContextManager/representatives/retrieval,
local+global attention and LRU. InfiniGen uses its author-derived threshold and
per-head selector, CPU historical KV and cross-layer query forecasts. Both
have independent Qwen hybrid-model adapters. They do not natively provide this
Qwen MTP integration; these results are adapted implementations, not untouched
upstream end-to-end systems.

The actual GGUF block-64 predictor shares the main embedding/output head.
Its prefix pairs token t with target pre-final-norm hidden t-1. It drafts four
tokens; the target verifies and commits only the accepted prefix plus its own
correction/bonus. Each token performs its own causal retrieval. InfiniGen
pipelines the individual forecasts after staging candidate K/V, with selector
i restricted to index[:base+i]. Rejected candidates never extend committed
history. InfLLM also restores local scores, block representatives, LRU state
and physical attention block ordering.

The compute boundary uses standalone Q8 CUDA arithmetic from a frozen qw3
source copy. Only the isolated copy changes: four-warps across verify widths,
explicit rounded accumulation/products, and per-token RMSNorm reductions avoid
batch-dependent rounding changes. The production qw3 source/binary and authors'
repositories remain unchanged. Recurrent and MLP CUDA graphs retain every
prefix state; retrieval and transfers execute outside graphs.

Validation: independent serial prefills, five-token verification crossing a
128-token block boundary, commits of 1/3/5 tokens, serial continuation,
logits/recurrent state and retrieval-state fingerprints all match exactly.
Compute graphs separately check outputs and every stored recurrent prefix.
All 24 formal MTP-on/off pairs generate exactly identical 128-token sequences.
This is implementation consistency evidence, not an agent accuracy evaluation.

## Comparison boundaries

InfLLM retains its small author-style working set: 128 initial + 4096 local +
16 x 128 retrieved tokens; recurrent layers preserve their recurrent states.
InfiniGen has a 2048-token per-query-head selection cap, alpha=4, partial ratio
0.2; it does not necessarily select exactly 2048 tokens. Both MTP drafters use
full GPU FP8 history at true positions, bounded by the validated 262144-position
range. Longer histories require an additional draft-context policy; they are
not claimed supported by this harness.

The previous native Q8/MTP=4 KVMem medians were 111.66/93.18/89.77/72.59 token/s
at 8K/32K/120K/240K. Its selection budget is 128K, its MTP follows the working
set, and native residual arithmetic is FP32 versus the adapters' HF/FLA BF16.
Input IDs, Q8 weight file, FP8 storage, hardware and timed token count align;
working set, kernels and residual arithmetic differ. Treat native numbers as
deployment references, not an isolated retrieval-algorithm comparison or an
accuracy equivalence claim. Raw native results remain in
`../decode_comparison_20260912/q8_mtp_20260913/export/`.

Early failed numerical checks and development probes are retained separately.
The first formal campaign stopped at 32K because upstream load(None) referenced
an uninitialized target_event when restoring an evicted block. Its complete
logs/results are retained in formal_failure_cache_reload_20260913. The adapter
now passes an explicit temporary destination, then reconstructs the saved buffer
order. A 32K eviction/rollback regression check passed. The entire formal matrix
was restarted after this fix; results were not filtered by speed or acceptance.
OSWorld pause/resume evidence is recorded in `resume_status.json` and the suite's
`throughput_pause_external_mtp_20260913/` directory.
