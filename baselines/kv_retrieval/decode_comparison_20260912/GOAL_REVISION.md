# Revised goal: matched Qwen3.8-27B decode comparison

User instruction, 2026-09-12: finish native upstream checks, then compare all
methods on Qwen3.8-27B in the same setting. Native OPT/Mistral results are
implementation checks, not the primary KVMem comparison.

1. Finish the unchanged InfiniGen OPT and InfLLM Mistral measurements. Preserve
   every repetition, including the observed InfLLM long-history variability.
   Bound the remaining NUMA check; do not postpone the Qwen work indefinitely.
2. Audit Qwen integrations against the authors' implementations. Use an isolated
   directory; do not modify the production qw3 engine. Retain recurrent layers,
   query normalization, RoPE, retrieval, prefetch and actual CPU KV transfers.
3. Prefer ordinary HF/PyTorch model execution for external methods. The previous
   Q8 adapter used qw3 projection kernels and is not an untouched upstream
   reproduction. Establish a same-weight/precision comparison and explicitly
   document every adaptation and numerical/backend difference.
4. Match input token IDs, hardware exclusivity, batch=1, output length, precision,
   warmup and decode timing. Report native small-window configurations and an
   aligned active-window configuration separately where feasible. No MTP in the
   primary method comparison; MTP can be a separate deployment reference.
5. Validate adapted computation and measure throughput including retrieval and
   transfers, excluding separately reported load, calibration and prefill.
   Do not infer agent accuracy from this microbenchmark.
6. Export raw data, configuration/provenance and a comparison report. Restore
   the paused OSWorld suite from task 002/KVMem, preserving completed results.

The active goal already requires a same-model comparison. This file records the
clarified model and upstream-backend constraints without closing that goal.

## User correction, 2026-09-13: deployed Q8_0 and MTP

The user clarified that the intended deployment comparison uses Q8_0 weights,
then explicitly requested enabling MTP and rerunning. The BF16/MTP=0 campaign
above is supplemental and does not fulfill the Q8/MTP comparison setting.
The corrected native deployment campaign is q8_mtp_20260913/; external Qwen
adapters currently lack MTP and are not labeled as matched MTP baselines.
