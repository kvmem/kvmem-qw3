# Qwen3.8-27B decode throughput comparison

All primary rows use the exact same original local BF16 safetensors, FP8 E4M3FN KV storage, batch=1, no MTP, no NVMe, exclusive RTX PRO 6000 Blackwell GPU, and CPU affinity on the GPU-local NUMA node. Model computation for external methods is HF/FLA/PyTorch; no qw3 projection library is loaded.

## Fixed 128-output measurement

Rates are median [minimum, maximum] tokens/s over three repetitions. The first output belongs to prefill; the timed interval covers the remaining 127 output tokens. Inputs are identical real DeepSWE history prefixes at each length, with the same current-query suffix. These are decode microbenchmarks, not task-success results.

| Method | 8K | 32K | 120K | 240K |
|---|---:|---:|---:|---:|
| HF full CPU KV | — | 16.69 [16.69, 16.71] | 7.69 [7.63, 7.81] | — |
| HF full GPU KV | 21.32 [21.28, 21.37] | 19.98 [19.92, 20.26] | 16.13 [16.03, 16.15] | 12.76 [12.75, 12.77] |
| InfiniGen · cap 2K | 11.74 [11.40, 12.47] | 9.85 [8.99, 9.86] | 8.44 [8.38, 8.64] | 8.24 [8.16, 8.33] |
| InfLLM · small window | 16.14 [16.06, 16.28] | 15.65 [15.54, 15.75] | 14.95 [14.82, 15.07] | 14.69 [14.63, 14.70] |
| KVMem · 128K + 128K | 23.98 [23.97, 24.15] | 23.66 [23.64, 23.82] | 22.49 [22.49, 22.73] | 22.45 [22.38, 22.47] |
| qw3 · full GPU KV | 24.43 [24.41, 24.45] | 24.13 [24.13, 24.13] | 23.00 [23.00, 23.02] | 21.66 [21.66, 21.67] |

## Attention budgets

InfLLM small uses 4096 recent + 128 initial + 16×128 retrieved tokens (~6.3K); its resident historical cache holds 32 blocks. InfiniGen small uses alpha=4, a 20% partial subspace and a 2048-token cap per query head; actual selected counts and H2D bytes are exported. KVMem keeps its 128K selection and 128K generation budgets, block=32, CPU history/index, and full explicit-query replay. Thus the small-window table matches model/precision/workload but deliberately preserves the authors’ small working sets. It does not establish equal task quality.

## Bounded equal-budget check

All methods have a 128K retrieval/selection budget on a 240K input. The exact per-method policy remains unchanged. InfiniGen’s cap is an upper bound, not a fixed selected count. Fixed 32 outputs (31 timed decode forwards); these results are kept separate from the 128-output table.

| Method | 240K |
|---|---:|
| InfiniGen · cap 128K | 2.48 (n=1) |
| InfLLM · 128K budget | 0.63 (n=1) |
| InfLLM · 128K budget, block 2048 | 4.04 (n=1) |
| InfLLM · 128K budget, exact CPU map optimization | 3.48 (n=1) |
| KVMem · 128K budget | 22.18 (n=1) |

InfLLM retains the author configuration’s 2x cache-to-retrieval-block ratio (991 retrieved / 1982 cached at block=128). A separate block=2048 configuration uses 62 retrieved / 124 cached blocks and 3968 recent + 128 initial tokens, for the same total 131072-token attention budget. Changing block granularity can affect selection/accuracy; both measurements are reported, not treated as numerically identical.

The exact CPU block-map optimization replaces nested list membership with sets and an ascending free-slot iterator. It preserves selected IDs, placement and copy order in the CPU tests, and all 32 generated IDs match the original in the real-model check. At unchanged block=128, throughput changes from 0.635 to 3.484 tokens/s (5.49x in this single check); prefill changes from 458.7 to 174.5 seconds. This is an isolated adapter optimization; the upstream checkout is unchanged. See block_map_model_validation.json.

## Interpretation and limits

- InfLLM and InfiniGen are Qwen architecture ports of the author methods. Their original end-to-end code natively targets Mistral/OPT, so those unchanged implementations were measured first in `../native_upstream_export/REPORT.md`. Do not call the Qwen port an unmodified upstream reproduction.
- Same-model system throughput can be compared, but the native engine and HF have different operator/launch paths and residual precision (native FP32 versus HF BF16). The HF full-GPU control exposes common execution overhead; the full-CPU control exposes the benefit of selective offloading within the HF pipeline.
- Historical KV storage is FP8 in all primary arms. InfLLM quantizes unrotated historical keys to preserve its position-remapping method; native/InfiniGen may quantize at a different positional boundary. InfLLM recent working views retain exact FP8-rounded values in FP16. These numerical/storage details must accompany accuracy claims.
- On this GQA model the per-query-head InfiniGen FP16 partial-key index is 1.195 times the full shared-head FP8 K+V size (8.965 vs 7.5 GiB at 240K). Thus this port does not establish a GPU-memory advantage over full FP8 KV; quantizing or restructuring its index would be a separate variant. See METHOD_AUDIT.md and infinigen_index_footprint.json.
- HF timing includes all per-token retrieval, CPU gathering, transfers, sampling and state updates. Native timing is first model token to server completion, which also includes final cache bookkeeping/serialization. Raw native decode counters are retained but are not used blindly: plain no-MTP KVMem counters can include initial selection/replay.
- Loading, SVD calibration, prefill, initial offload and initial KVMem query selection/replay are outside decode throughput and reported separately. This table does not measure Claude Code summary stalls or private guided-query generation.
- Important rollback observation: repeating an identical 240K raw completion prompt retracts the previous generated suffix. KVMem hit no_reusable_checkpoint twice, with 109.575 and 108.352 seconds of cold prefill. Those original prefill and request-wall measurements are retained in runs.csv and rollback_checkpoint_observation.json. The decode table cannot be read as an end-to-end request-latency claim, nor does this establish the frequency of that case in append-only agent traffic.
- The 240K history fits the tested full-GPU controls. CPU-offload controls intentionally force CPU history; this is not evidence that full GPU KV is infeasible on this hardware.
- All trials, tuning and validity checks are preserved. NUMA affinity was selected after unbound upstream InfLLM runs showed variability; the original slow measurements remain in the auxiliary export.

## Provenance and validation

The native BF16 compatibility directory adds only dense-format loader metadata and symlinks the original safetensors. No weight values or production engine code are changed. See `weight_identity_audit.json`, `native_bf16_container_manifest.json`, `formal_jobs.json`, `bf16_calibration_validation.json` and the raw per-arm reports. InfiniGen full-selection logits and CPU KV match the dense control exactly across eight real-model decode steps; graph checks verify both outputs and recurrent state.

Sources: [InfLLM](https://github.com/thunlp/InfLLM/tree/12b70798f56e56ebb23c53c7018091a3f540a028), [InfiniGen](https://github.com/snu-comparch/InfiniGen/tree/3754f52d131c30ab7a310e3c4cf356e8e3326ac3).

Data: `export/runs.csv`, `export/summary.csv`, `export/decode_steps.csv`; bounded budget check: `matched_budget_export/`.
