# Paper pre-answer latency results

Date: 2026-08-04

The filled LaTeX table is in `docs/paper_utility_efficiency_table.tex`.  The
machine-readable aggregate is:

```text
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_20260803/summary.json
```

## Results

| Benchmark | Full Context | Sliding Window | Compact-only | Compact+RAG | KVMem |
|---|---:|---:|---:|---:|---:|
| LongMemEval-S | 0.81 | 42.29 | 68.88 (3.50) | 116.13 (50.75) | 1.68 |
| MemoryAgentBench (>256K) | -- | 0.22 | 344.38 (1.22) | 357.81 (14.66) | 4.28 |
| AgentLongBench (<=256K) | 0.08 | 41.43 | 326.80 (1.32) | 380.94 (55.45) | 0.63 |
| AgentLongBench (512K) | -- | 39.12 | 1317.61 (68.86) | 1449.19 (200.44) | 1.07 |
| AgentLongBench (1M) | -- | 39.12 | -- (17.57) | -- (71.82) | 1.37 |

All values are seconds.  Parentheses exclude compact-summary generation.
Full Context is unavailable where the complete history exceeds the model and
resident KV capacity.  LongMemEval-M remains unfilled because there is no
complete frozen utility result to pair with a latency configuration.

## Measurement boundary and aggregation

- The boundary starts at query-dependent context preparation and ends at the
  first non-empty model token.
- Cold source-history ingestion, server startup, subsequent answer decoding,
  grading and result persistence are excluded.
- AgentLongBench uses the same deterministic P25/P50/P75 stable IDs for every
  applicable method and reports the arithmetic mean over three samples.
- MemoryAgentBench uses the exact utility cohort with canonical archive length
  greater than 262,144 tokens: 30 contexts and 1,316 questions, aggregated
  question-wise.
- The MemoryAgentBench KVMem artifacts predate client-side first-token logging.
  Its 4.28-second value is reconstructed as
  `wall_s - decode_s + decode_s / decoded_tokens`; all newer AgentLongBench
  KVMem cells use direct client-observed TTFT.
- AgentLongBench 1M summary-tail artifacts retain query/retrieval timings but
  not the original DeepSeek compact-summary generation timing.  Consequently,
  only the parenthesized Compact values are reported instead of inventing a
  non-amortized main value.

## Controlled KVMem measurements

| Slice | Per-sample TTFT (s) | Mean (s) | Scorer fallback |
|---|---|---:|---:|
| AgentLongBench <=256K, Full Context | 0.090, 0.091, 0.056 | 0.079 | N/A (no reselection) |
| AgentLongBench <=256K, KVMem | 0.513, 0.723, 0.669 | 0.635 | 0 |
| AgentLongBench 512K, KVMem | 1.075, 1.142, 0.981 | 1.066 | 0 |
| AgentLongBench 1M, KVMem | 1.452, 1.222, 1.424 | 1.366 | 0 |

The Full Context control uses an all-fit 224K context budget plus a 32K
generation reserve solely to hold the complete <=256K histories.  Final
reselection is disabled and the trace contains neither pressure selection nor
query-conditioned retrieval.  Its GPU ratio is 0.55 because an FP16 256K page
pool cannot fit under the 0.50 process ceiling.  This does not change any
32K-budget utility configuration.

## Raw artifacts

```text
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_20260803/alb_le256_full_context.jsonl
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_20260803/alb_le256_kvmem.jsonl
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_20260803/alb_512k_kvmem.jsonl
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_20260803/alb_1m_kvmem_v3.jsonl
/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802
/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full
```
