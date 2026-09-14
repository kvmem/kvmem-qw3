# InfiniGen Qwen/MTP bottleneck attribution

Keep the production qw3 binary, upstream repositories and completed benchmark
artifacts unchanged. Pause OSWorld at the current arm boundary and restore its
frozen queue after GPU work. Qwen3.8-27B Q8_0, FP8 history, same GPU, NUMA affinity,
input prefixes, small retrieval budgets, MTP=0/4 and exact target semantics.

1. Tune 2/4/8 CPU threads at 32K; record the actual prefetch worker thread count.
2. Compare unprofiled full runs with same-output diagnostic variants: cached
   partial query; cached selected IDs; GPU-resident identical historical KV;
   and cached IDs plus GPU history. These are oracle/capacity diagnostics,
   never presented as deployable baselines or accuracy results.
3. Collect forecast, selector/index synchronization, CPU sort/dedup/gather,
   H2D, GPU reconstruction and consumer wait timings. Separate instrumented
   profiles from performance measurements; asynchronous phase totals overlap.
4. Compare against InfLLM using the same executor and arithmetic. Reset exact
   initial cache/recurrent/predictor states between repetitions so prefill need
   not be repeated; check generated token IDs against the prior fresh-prefill
   formal results. Keep rejected speculative work in timing and profiling.
5. Report measurable implementation savings and structural work separately.
   An oracle elimination is a bound, not proof that the whole eliminated
   interval is inherent to the method. Preserve all data and resume OSWorld.
