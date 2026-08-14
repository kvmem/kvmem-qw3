# Paper pre-answer latency results

Date: 2026-08-04

> Superseded: these historical estimates have been replaced by the controlled
> current-machine rerun in `docs/paper_pre_answer_latency_results_20260812.md`.
> Do not copy latency values from this file into the paper table.

The filled LaTeX table is in `docs/paper_utility_efficiency_table.tex`.  The
machine-readable aggregate is:

```text
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_20260803/summary.json
```

## Results

| Benchmark | Full Context | Sliding Window | Compact-only | Compact+RAG | KVMem |
|---|---:|---:|---:|---:|---:|
| LongMemEval-S | 0.17* | 8.83* | 14.39 (0.73)* | 24.26 (10.60)* | 1.68 |
| MemoryAgentBench (>256K) | -- | 8.63 | 344.38 (1.22) | 357.81 (14.66) | 4.28 |
| AgentLongBench (<=256K) | 0.08 | 8.65* | 68.26 (0.27)* | 79.57 (11.58)* | 0.63 |
| AgentLongBench (512K) | -- | 8.17* | 275.22 (14.38)* | 302.71 (41.87)* | 0.37† |
| AgentLongBench (1M) | -- | 8.17* | -- (3.67)* | -- (15.00)* | 0.41† |

All values are seconds.  Parentheses exclude compact-summary generation.
Full Context is unavailable where the complete history exceeds the model and
resident KV capacity.  LongMemEval-M remains unfilled because there is no
complete frozen utility result to pair with a latency configuration.

`*` Historical baseline values are normalized estimates targeting the current
NVIDIA RTX PRO 6000 Blackwell Server Edition and native QW3 backend.  The
combined factor is `2.370205 * 2.019826 = 4.787403`: the first factor converts
the matched 32K llama.cpp workload from A40 (`39.117871 s`) to RTX PRO 6000
(`16.504 s`), and the second converts llama.cpp to native QW3 on that GPU
(`16.504 / 8.171 s`).  The table divides the complete historical baseline
latency by this factor.  This deliberately simple estimate also scales the
smaller CPU/tokenizer/retrieval fraction and must not be presented as a rerun.
LongMemEval-S uses the same conversion as a legacy estimate because its
per-sample hardware provenance is incomplete.  Raw values remain available in
the machine-readable aggregate.

`†` The AgentLongBench-512K and 1M KVMem cells use the newer K64/B32 and
K100/B128 utility configurations.  Their 0.37 s and 0.41 s values are estimates
from the arithmetic mean of conserved server-side timing over all 100 and 50
utility samples; see the LaTeX table footnote for the included components.

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
- Sliding Window uses a cold-window boundary: the selected 32K text window is
  prefilled from scratch for every independent question, with no prefix-cache
  hit credited.  The MemoryAgentBench value is reconstructed question-wise as
  `context cold-window warmup TTFT + observed question-suffix TTFT`, yielding
  `8.4118 + 0.2172 = 8.6290` seconds.  The common recent-window selection is
  not charged per question because the frozen utility runner selects it once
  per context and it is independent of the question.
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
