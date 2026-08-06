# Paper Pre-answer Latency Completion Plan

Date: 2026-08-03

Status: completed on 2026-08-04 for every recoverable table cell. Results and
limitations are recorded in `docs/paper_pre_answer_latency_results_20260804.md`;
the filled LaTeX source is `docs/paper_utility_efficiency_table.tex`.

## Goal and publication contract

Fill the missing pre-answer latency cells in the paper's controlled
long-history benchmark table. Every latency result must use the same frozen
method configuration that produced the utility value in the corresponding
table cell. Active-context budgets are therefore method- and benchmark-specific;
they are not uniformly 32K.

The main latency boundary is:

```text
pre-answer latency
= wall time from the start of query-dependent context preparation
  until the client receives the first non-empty reasoning/content token
```

It includes, when applicable:

- sliding-window selection and prompt preparation;
- compact-summary construction for the non-parenthesized Compact values;
- RAG block/query encoding, ranking, selected-text loading and prompt assembly;
- final-query tokenization/prefill;
- KVMem scoring, stage-out exposed on the critical path, stage-in,
  materialization/re-RoPE, semantic commit and query replay;
- scheduling and the first-token generation step exposed to the client.

It excludes:

- final-answer decoding after the first non-empty token;
- grading and result persistence;
- cold ingestion of the complete source history for Full Context and KVMem;
- model/server startup and one-time kernel warm-up.

For Compact-only and Compact+RAG, the main value charges the complete
non-amortized summary-generation cost. The parenthesized value excludes summary
generation and reports the steady-state query cost. This asymmetry must remain
explicit in the caption and table note.

KVMem's internal operation-only value remains useful as a breakdown, but it is
not the main table value:

```text
kvmem_operation_ms = post_semantic_ms + post_query_replay_ms
```

The main KVMem cell must use the client-observed pre-answer latency from a
frozen history state. It must not publish `kvmem_operation_ms` as though it were
the complete TTFT.

## Revised paper caption

Use wording equivalent to the following when the table source is updated:

```text
Utility and efficiency results across controlled long-history benchmarks.
Higher values are better for utility metrics, while lower values are better
for pre-answer latency. Each method is evaluated using the frozen configuration
that produced its reported utility result; active-context budgets are
method- and benchmark-specific and are listed in the appendix. Full Context is
reported only when the complete history fits the model and KV-cache capacity.
Pre-answer latency is measured from the start of query-dependent context
preparation to the first non-empty model token, including method-specific
retrieval and final-query input processing, while excluding subsequent answer
decoding and cold source-history ingestion. For Compact-only and Compact+RAG,
values in parentheses exclude compact-summary generation.
```

The old sentence claiming that every non-Full method uses a 32K active context
must be removed.

## Frozen utility configurations

Latency runs must reproduce the utility configuration for the matching table
cell. At minimum, save model revision, context limit, active budget, generation
reserve, block/sub-block size, retrieval method, KV/index/query dtype, thinking
mode, sampling parameters, prefix/query-replay policy and tier placement.

The currently identified KVMem configurations are:

| Benchmark slice | Frozen KVMem utility configuration |
|---|---|
| LongMemEval-S | K=32K; G=32K; block=32; key-direction-adaptive; FP16 KV/index/query; chunk=2K; thinking=4K |
| AgentLongBench <=256K | K=32K; G=32K; block=32; mean-k; FP16 KV; chunk=2K; thinking=4K |
| MemoryAgentBench >256K | K=200K; G=32K; block=512; key-direction-adaptive 1/2/4 prototypes; FP8 KV with FP16 index/query; chunk=2K; CPU index; query replay; immutable KV; all three transfer optimizations |
| AgentLongBench 512K | K=224K; G=32K; block=32; mean-k; FP16 KV; chunk=2K; thinking; query replay; immutable KV |
| AgentLongBench 1M | K=224K; G=32K; block=512 with 32-token scoring slices; key-direction-fixed4 MaxSim; FP8 KV with FP16 index/query; chunk=2K; thinking=8K; query replay; immutable KV |

The baseline configurations must likewise be read from the exact utility
artifacts rather than replaced with a new common 32K setting. Record the actual
active prompt tokens per sample because Compact and Compact+RAG prompt lengths
are variable.

If any latency run differs from its frozen utility configuration, it belongs in
an ablation and must not fill this table.

## Table coverage

LongMemEval-S currently has candidate latency values, but their raw provenance
and the revised end-to-end boundary must be audited before treating the row as
final:

| Method | Candidate pre-answer latency |
|---|---:|
| Full Context | 0.81 s |
| Sliding Window | 42.29 s |
| Compact-only | 68.88 s (3.50 s) |
| Compact+RAG | 116.13 s (50.75 s) |
| KVMem | 1.68 s |

LongMemEval-M remains out of scope until a complete utility result exists. Keep
its latency cells as `--` or `TBD`.

The remaining work is:

| Benchmark slice | Full Context | Sliding Window | Compact-only | Compact+RAG | KVMem | Action |
|---|---|---|---|---|---|---|
| MemoryAgentBench (>256K) | Keep `--` | Missing | Missing | Missing | Missing | Reuse raw artifacts plus targeted timing-only reruns |
| AgentLongBench (<=256K) | Missing | Missing | Missing | Missing | Missing | Three-sample controlled measurement |
| AgentLongBench (512K) | Keep `--` | Missing | Missing | Missing | Missing | Three-sample controlled measurement |
| AgentLongBench (1M) | Keep `--` | Missing | Missing | Missing | Missing | Three-sample controlled measurement |

Full Context must not be measured for a slice above the model's complete-history
capacity. In particular, MemoryAgentBench >256K, AgentLongBench-512K and
AgentLongBench-1M retain `--` in that column.

## Cache state and warm-up policy

The history state must be ready before the measured final question:

- Full Context: complete lossless history KV is resident/restorable;
- Sliding Window: the method's frozen recent window is selected as specified by
  its utility runner; whether its KV is prefetched must match that runner;
- Compact: the parenthesized measurement starts from the frozen completed
  summary; the main value adds the separately measured full summary-generation
  cost;
- Compact+RAG: starts retrieval from the same completed compact state;
- KVMem: starts from the frozen KVMem history state with the same CPU/NVMe/GPU
  tier policy as the utility run.

Perform one unrelated short synthetic request to warm model kernels. Never use
the measured sample's final query as the warm-up: doing so warms selected KVMem
blocks, RAG data and prefix-cache pages and changes the measured operation.

Every retry/repetition must restore the same pre-query frozen state. The answer
from a previous trial must not be appended to the next trial.

## MemoryAgentBench >256K

Use exactly the same population as the utility table:

- canonical archive length strictly greater than 262,144 tokens;
- 30 contexts;
- 1,316 questions;
- question-weighted arithmetic mean;
- match contexts by `(split, source, dataset_row)`, not directory ordinal.

Relevant artifacts:

```text
/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802
/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802/comparison_over256k.json
/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full
```

Do not use the existing `archive_performance_summary.json` as the sole source:
it is a partial 77-context/2,847-question snapshot. The complete raw directory
contains all 146 contexts, and all selected 30 contexts/1,316 questions can be
parsed from their row logs.

The existing artifacts are sufficient for utility and for several latency
components, but not for every final table value:

- Compact states contain per-round selection and full generation wall times;
- baseline answer JSONL contains prefix-cached final-query TTFT;
- RAG states contain block embedding, batched query embedding and batched
  ranking time, but not separately attributable per-question selected-text
  loading/prompt assembly;
- Sliding selection time was measured while finding one common window for all
  questions in a context, not as an independent-question operation;
- KVMem row logs contain semantic/replay accounting, but the archived answer
  records do not contain client time to first token.

Therefore this row is not “offline extraction only.” Reuse the existing
compaction and utility artifacts, but run targeted timing-only measurements for
the missing independent-question boundaries. Do not regenerate or rejudge
utility answers.

### Sliding Window

```text
latency = independent window selection/preparation + final-query TTFT
```

Do not charge the old multi-question common-window search time to every question.
Reproduce the exact frozen sliding-window selection rule independently for each
question, or explicitly preserve the utility runner's shared-window policy and
measure only the work actually performed once. The selected window and active
prompt must remain identical to the utility artifact.

### Compact-only

```text
main value    = complete non-amortized summary generation + final-query TTFT
parenthesized = final-query TTFT
```

Use the full wall time of every compact round. Include segment selection and
prompt preparation in summary construction. Charge the complete chain to each
independent question; do not divide it by the number of questions sharing the
context.

### Compact+RAG

```text
main value    = complete compaction + independent retrieval + final-query TTFT
parenthesized = independent retrieval + final-query TTFT
```

Retrieval includes block/query encoding required by the frozen implementation,
ranking, selected-text loading and prompt construction. Preserve whether block
embeddings are built per context or per question in the utility implementation;
do not silently replace a batched timing with a per-question timing or vice
versa. Report both charged and amortized preprocessing in the appendix if the
implementation shares work across questions.

### KVMem

Restore/rebuild the frozen context once, branch every question independently,
and record client TTFT from request submission to the first non-empty token.
Also retain:

```text
query_prefill_ms
post_semantic_ms
post_query_replay_ms
post_other_ms
kvmem_operation_ms = post_semantic_ms + post_query_replay_ms
```

The main cell is the client-observed pre-answer latency, not
`kvmem_operation_ms`.

## AgentLongBench representative measurement

Use three samples per length slice. This is a representative latency estimate,
not a full-population latency evaluation.

For each slice:

1. Render each sample with the canonical worker used by its frozen utility run.
2. Tokenize the history without the final question using qw3 `/tokenize`.
3. Sort the frozen utility population by rendered history-token length.
4. Select the samples closest to P25, P50 and P75, with stable sample ID as the
   deterministic tie-breaker.
5. Persist the three IDs and their history/query token counts before launching
   a method.
6. Run every applicable method on exactly the same three IDs.
7. Warm kernels only with an unrelated synthetic request, then restore the
   clean pre-query state and perform one recorded run.
8. If a run fails, retry the same sample from the clean state; do not replace
   it with another sample.
9. Report the arithmetic mean across the three samples and retain all raw
   per-sample values.

| Slice | Frozen population | Samples | Methods |
|---|---:|---:|---|
| AgentLongBench <=256K | 250 | 3 | Full Context, Sliding Window, Compact-only, Compact+RAG, KVMem |
| AgentLongBench 512K | 100 | 3 | Sliding Window, Compact-only, Compact+RAG, KVMem |
| AgentLongBench 1M | 50 | 3 | Sliding Window, Compact-only, Compact+RAG, KVMem |

This is 39 method-sample measurements:

```text
3 * 5 + 3 * 4 + 3 * 4 = 39
```

Compact-only and Compact+RAG may reuse an identical generated summary when the
frozen compact configuration and input are identical. Both main values must
still include the recorded full, non-amortized summary-generation time.

## Unified runner and timing requirements

The runner must persist timing beside the sample ID. It must not depend on
positional association with a shared server log. Either return native timing in
the response or write a trace-tagged structured sidecar for each request.

A minimal JSONL record is:

```json
{
  "sample_id": "...",
  "slice": "le256k|512k|1m|memoryagentbench_gt256k",
  "method": "full|sliding|compact|compact-rag|kvmem",
  "history_tokens": 0,
  "query_tokens": 0,
  "active_prompt_tokens": 0,
  "active_context_budget": 0,
  "generation_reserve": 0,
  "kvmem_block_tokens": 0,
  "selection_or_retrieval_ms": 0.0,
  "compaction_ms": 0.0,
  "query_prefill_ms": 0.0,
  "post_semantic_ms": 0.0,
  "post_query_replay_ms": 0.0,
  "post_other_ms": 0.0,
  "kvmem_operation_ms": 0.0,
  "client_ttft_ms": 0.0,
  "pre_answer_latency_ms": 0.0,
  "cache_state": "clean_frozen_branch",
  "attempt": 1,
  "status": "completed"
}
```

For every run also save:

- exact command, environment overrides and configuration;
- code and model revision;
- KV/index/query dtype;
- active-context and generation budgets;
- scorer mode and fallback count;
- CPU/NVMe/GPU tier placement and cache state;
- exact selected sample manifest;
- client/server clock source and raw structured timing.

Required invariants:

```text
pre_answer_latency_ms == client_ttft_ms
kvmem_operation_ms == post_semantic_ms + post_query_replay_ms
```

Native phase sums are a diagnostic and should agree with their enclosing server
interval within the timer precision. They need not exactly equal client TTFT
because client TTFT also includes API/scheduling and first-token delivery.

## One-sample smoke gate

Before the 39 measurements, run one <=256K sample through all five methods and
verify:

- all methods use the same stable sample and canonical rendered question;
- every method reproduces its frozen utility configuration;
- client TTFT stops at the first non-empty reasoning/content token;
- no measured final query was used as warm-up;
- KVMem native timing is trace-tagged and associated with the correct sample;
- KVMem scorer fallback count is zero unless the frozen utility run also used
  that fallback;
- no answer decoding after the first token is included;
- the main and parenthesized Compact values follow the formulas above;
- output JSONL contains every required field and the effective config.

Only after this gate passes should the full latency collection begin.

## Existing logs

Old AgentLongBench logs remain sanity checks only:

| Slice | Recovered value | Limitation |
|---|---:|---|
| <=256K | 0.69 s (`n=250`) | Old configuration has no query replay; lower bound only |
| 512K | 4.96 s (`n=56`) | Only 56 samples align from surviving logs |
| 1M | 4.71 s (`n=50`) | Different block size, scorer, placement and code revision |

These values must not replace the controlled measurements or be presented as a
controlled context-length scaling curve.

## Completion checklist

- [ ] Update the paper caption to remove the uniform-32K claim.
- [ ] Add the frozen per-cell configurations and active budgets to the appendix.
- [ ] Audit the five existing LongMemEval-S latency values against the revised boundary.
- [ ] Implement trace-tagged per-sample KVMem timing and client TTFT persistence.
- [ ] Pass the one-sample, five-method smoke gate.
- [ ] Freeze P25/P50/P75 AgentLongBench IDs and rendered token lengths.
- [ ] Complete targeted MemoryAgentBench timing for the same 30 contexts/1,316 questions.
- [ ] Measure five methods on the three AgentLongBench <=256K samples.
- [ ] Measure four methods on the three AgentLongBench-512K samples.
- [ ] Measure four methods on the three AgentLongBench-1M samples.
- [ ] Verify that no Full Context run is launched above complete-history capacity.
- [ ] Verify every main KVMem cell uses client TTFT, with semantic/replay retained only as breakdown.
- [ ] Verify Compact main values include full non-amortized compaction and parentheses exclude it.
- [ ] Report `n=3` for AgentLongBench representative latency estimates.
