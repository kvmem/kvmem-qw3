# Paper pre-answer latency results (current-machine rerun)

Date: 2026-08-12

All non-TBD latency cells were measured on the same NVIDIA RTX PRO 6000
Blackwell Server Edition with the same native `qw3` binary and
`Qwen3.6-27B-Q8_0.gguf` model. The final machine-readable validation status is
`complete`.

## Results

| Benchmark | Full Context | Sliding Window | Compact-only | Compact+RAG | KVMem |
|---|---:|---:|---:|---:|---:|
| LongMemEval-S | 0.30 | 0.19 | 18.92 (2.91) | 26.63 (10.63) | 0.48 |
| MemoryAgentBench (>256K) | -- | 0.26 | 85.87 (0.57) | 106.02 (20.72) | 1.81 |
| AgentLongBench (<=256K) | 0.17 | 0.11 | 97.04 (0.45) | 111.39 (14.80) | 0.38 |
| AgentLongBench (512K) | -- | 0.20 | 246.49 (6.36) | 263.70 (23.58) | 0.62 |
| AgentLongBench (1M) | -- | 0.26 | 380.19 (3.05) | 416.38 (39.24) | 0.73 |

All values are seconds. Parentheses exclude the boundary compact-summary
generation but still include retrieval/materialization and final-context
prefill. LongMemEval-M remains TBD because there is no complete matched utility
and baseline result.

## Measurement boundary

- Aggregation: one deterministic P50-length representative per benchmark
  slice. Utility values in the paper table still use the full benchmark cohort.
- Start: arrival of the final question when the source history has already been
  maintained and the active context is full.
- End: first non-empty model token.
- Excluded: earlier history ingestion/maintenance, subsequent answer decoding,
  grading, and result persistence.
- Full Context: final-query continuation TTFT from resident full history.
- Sliding Window: final-query continuation TTFT from the already-resident
  recent window; history-window selection/maintenance is excluded.
- Compact-only: one boundary compaction, prompt assembly/re-prefill, and final
  query TTFT.
- Compact+RAG: one boundary compaction, query-conditioned retrieval, prompt
  assembly/re-prefill, and final query TTFT.
- KVMem: final semantic scoring, selected-block materialization, query replay,
  short final-query prefill, and first-token decode.

## Controlled capacities

| Slice | Active cap | KVMem configuration |
|---|---:|---|
| LongMemEval-S | 32K | K32/B32, Adaptive Mean-K, FP16 KV |
| MemoryAgentBench (>256K) | 64K | K64/B64, Adaptive Mean-K, FP8 KV, CPU archive |
| AgentLongBench (<=256K) | 32K | K32/B32, Mean-K, FP16 KV |
| AgentLongBench (512K) | 64K | K64/B32, Adaptive Mean-K, FP8 KV |
| AgentLongBench (1M) | 100K | K100/B128, Adaptive Mean-K, FP8 KV |

All KVMem rows use a 32K generation reserve, 2K prefill chunks, immutable K,
query replay, MTP chain 4, CPU-only lower-tier backing, and stage-out,
stage-in, and packed-transfer optimizations. Sliding Window uses the same active
cap as KVMem. Compact-only and Compact+RAG enforce that same final prompt cap,
use no raw tail and no RAG metadata headers, and let the summary consume part
of the cap before RAG fills the remainder.

## KVMem final-query breakdown

| Slice | TTFT | Query prefill | Semantic selection | Query replay | Boundary history replay |
|---|---:|---:|---:|---:|---:|
| LongMemEval-S | 0.482 | 0.191 | 0.198 | 0.088 | 27 tokens |
| AgentLongBench (<=256K) | 0.384 | 0.150 | 0.180 | 0.047 | 5 tokens |
| AgentLongBench (512K) | 0.617 | 0.221 | 0.305 | 0.084 | 15 tokens |
| AgentLongBench (1M) | 0.728 | 0.223 | 0.405 | 0.089 | 62 tokens |

The frozen raw-token session now splits at the immediately preceding physical
block boundary. Therefore the final measured request replays fewer than one
physical block of history. The validator rejects any result that violates this
condition; this prevents a distant independently-tokenizable text boundary from
silently inflating query TTFT.

MemoryAgentBench uses its archive query path and reports 1.806 seconds from
query arrival through its first non-empty native callback. Archive construction
and attach are excluded.

## Reproducibility artifacts

```text
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812/summary.json
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812/summary.md
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812/latency_rows.tex
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812/baseline_validation.json
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812/kvmem_validation.json
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812/final_all_validation.json
```

Reproduction entry points:

```text
scripts/kvmem_eval/supervise_paper_latency_qw3_all.sh
scripts/kvmem_eval/supervise_paper_latency_qw3_current_machine.sh
scripts/kvmem_eval/supervise_paper_latency_kvmem_long_current_machine.sh
scripts/kvmem_eval/validate_paper_latency_current.py
scripts/kvmem_eval/summarize_paper_pre_answer_latency_current.py
```
