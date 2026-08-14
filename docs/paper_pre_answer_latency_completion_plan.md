# Paper Pre-answer Latency Completion Plan

Date: 2026-08-12

Status: completed. The validated current-machine results are in
`docs/paper_pre_answer_latency_results_20260812.md`; machine-readable artifacts
are under `/data/chaidi/kvmem_eval/results/`
`paper_pre_answer_latency_qw3_current_20260812`. The 2026-08-04 values are
historical estimates and must not be copied into the final table.

All formal latency cells in this rerun use the native `qw3` server from this
repository. vLLM runs and historical provider-side timings are diagnostic only
and are excluded from the paper table. The plain-baseline and KVMem passes
record their binary SHA256 and GGUF path; final validation requires both
identities to match.

The canonical resumable launch is:

```bash
tmux new-session -d -s paper_latency_qw3_all_20260812 \
  'cd /home/chaidi/qw3 && bash scripts/kvmem_eval/supervise_paper_latency_qw3_all.sh'
```

Formal artifacts are isolated under:

```text
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812/
```

## Goal and publication contract

Fill the missing pre-answer latency cells in the paper's controlled
long-history benchmark table. Every latency result must use the same frozen
method configuration that produced the utility value in the corresponding
table cell. Active-context budgets are therefore method- and benchmark-specific;
they are not uniformly 32K.

The final-query boundary assumes the method has already maintained the source
history and that its current active context is full. The main latency is:

```text
pre-answer latency
= wall time from final-query arrival
  until the client receives the first non-empty reasoning/content token
```

It includes, when applicable:

- exactly one final full-window compact-summary operation for the
  non-parenthesized Compact values;
- RAG query encoding, scoring/ranking, selected-text loading and prompt
  assembly;
- re-prefill of the newly compacted summary/context;
- final-query tokenization/prefill;
- KVMem scoring, stage-out exposed on the critical path, stage-in,
  materialization/re-RoPE, semantic commit and query replay;
- scheduling and the first-token generation step exposed to the client.

It excludes:

- final-answer decoding after the first non-empty token;
- grading and result persistence;
- cold ingestion of the complete source history for Full Context and KVMem;
- Sliding suffix selection, resident-window prefill and other history
  maintenance completed before the final query;
- earlier iterative Compact rounds completed while maintaining history;
- RAG corpus chunking and block-embedding/index construction completed while
  maintaining history;
- model/server startup and one-time kernel warm-up.

For Compact-only and Compact+RAG, the main value charges one complete
query-boundary compaction. It does not charge every earlier maintenance round
again. The parenthesized value excludes that final boundary compaction but
retains query-dependent RAG, final prompt materialization/re-prefill and TTFT.

The exact per-method sums are:

```text
Full Context
  = final-query continuation TTFT from resident full-history KV

Sliding Window
  = final-query continuation TTFT from resident selected-window KV

Compact-only
  = final boundary compaction
  + final compact-context prompt materialization/re-prefill
  + final-query TTFT

Compact+RAG
  = final boundary compaction
  + query embedding + scoring/top-k
  + selected-text/prompt materialization
  + final compact+retrieved-context re-prefill
  + final-query TTFT

KVMem
  = semantic scoring/reselection
  + exposed stage-out/stage-in/materialization/re-RoPE
  + query replay/prefill
  + first-token generation
```

KVMem's internal operation-only value remains useful as a breakdown, but it is
not the main table value:

```text
kvmem_operation_ms = post_semantic_ms + post_query_replay_ms
```

The main KVMem cell must use the client-observed pre-answer latency from a
frozen history state. It must not publish `kvmem_operation_ms` as though it were
the complete TTFT.

## Server-side TTFT instrumentation

The qw3 HTTP server now records a monotonic timestamp at the beginning of every
`/v1/chat/completions` or `/v1/completions` handler and explicitly reports:

| Field | Boundary | Intended use |
|---|---|---|
| `server_ttft_sec` | HTTP handler entry to the first non-empty model piece | Primary cross-dataset server TTFT |
| `engine_ttft_sec` | Actual engine invocation to the first non-empty model piece | Separates JSON/render/tokenize/queue overhead from inference |
| `response_ttft_sec` | HTTP handler entry to the first successfully written non-empty streamed reasoning/content/tool delta | Client-visible streaming boundary |
| `request_total_sec` | HTTP handler entry to response completion/build | Diagnostic total; includes answer decoding |

For non-streaming Chat Completions, `server_ttft_sec` and `engine_ttft_sec` are
still measured at the internal first-token callback, while
`response_ttft_sec` is `null` because the client does not receive a partial
response. For prefill-only requests (`max_tokens=0`) all TTFT fields are `null`
because no output token exists. `/v1/completions` retains its legacy
engine-relative `first_token_sec` field for compatibility.

Non-streaming responses expose these values in the top-level `timing` object.
Streaming Chat Completions attach the same object to the final chunk (the chunk
whose choice has a non-null `finish_reason`), independently of
`stream_options.include_usage`. Every completed request also emits one
machine-parseable server-log line:

```text
[qw3-server-ttft] rid=42 route=plain stream=1 model_token_seen=1 server_ttft_ms=412.300000 engine_ttft_ms=398.100000 visible_output_seen=1 response_ttft_ms=412.650000 request_total_ms=1832.700000
```

Use `server_ttft_sec` as the paper measurement when the benchmark uses a
frozen/session request for the final question. The metric includes every piece
of work submitted in that request. Consequently, if a runner sends the complete
history and the final question together, `server_ttft_sec` correctly includes
the complete cold-history prefill and is **not** the table's amortized
pre-answer latency. To exclude cold history ingestion, first ingest/freeze the
history in an unmeasured request, then time a separate final-query request that
restores or branches from that state.

The plain dense-prefix measurements run with MTP disabled because native qw3
prefix-cache v1 intentionally cold-prefills MTP requests. KVMem latency keeps
the frozen MTP setting used by its utility configuration. Since the metric
ends at the first token, this serving constraint and the exact server flags are
reported with the latency artifacts rather than hidden behind an estimated
conversion factor.

## Revised paper caption

Use wording equivalent to the following when the table source is updated:

```text
Utility and efficiency results across controlled long-history benchmarks.
Higher values are better for utility metrics, while lower values are better
for pre-answer latency. Each method is evaluated using the frozen configuration
that produced its reported utility result; active-context budgets are
method- and benchmark-specific and are listed in the appendix. Full Context is
reported only when the complete history fits the model and KV-cache capacity.
Pre-answer latency is measured from final-query arrival, with the source
history already maintained and the current active context full, to the first
non-empty model token. It includes query-triggered retrieval, exactly one
boundary compaction when applicable, final-context re-prefill and final-query
processing, while excluding subsequent answer decoding and all earlier history
maintenance. For Compact-only and Compact+RAG, values in parentheses exclude
the final boundary compaction.
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
| MemoryAgentBench >256K | K=64K; G=32K; block=64; key-direction-adaptive 1/2/4 prototypes; FP8 KV with FP16 index/query; 2K semantic-chunk reselection; sink=512; recent=0; GPU index; query replay; immutable KV (refresh=8); CPU-only archive backing; all three transfer optimizations |
| AgentLongBench 512K | K=64K; G=32K; block=32; key-direction-adaptive; FP8 KV with FP16 index/query; 2K semantic-chunk reselection; sink=512; recent=0; thinking=8K; query replay; immutable KV; CPU-only backing |
| AgentLongBench 1M | K=100K; G=32K; block=128 with 32-token scoring slices; key-direction-adaptive; FP8 KV with FP16 index/query; 2K semantic-chunk reselection; sink=512; recent=0; thinking=8K; query replay; immutable KV; CPU-only backing |

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
| AgentLongBench (<=256K) | Missing | Missing | Missing | Missing | Missing | One P50-length representative |
| AgentLongBench (512K) | Keep `--` | Missing | Missing | Missing | Missing | One P50-length representative |
| AgentLongBench (1M) | Keep `--` | Missing | Missing | Missing | Missing | One P50-length representative |

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

Use one sample per length slice: the sample closest to the canonical rendered
history-length median (P50). This is a representative point latency
measurement, not a population latency estimate. The main table must label the
latency cells as `n=1, P50-length representative`; it must not describe them as
means over the full benchmark.

The original frozen P25/P50/P75 candidates and their canonical history/query
token counts are persisted in
`scripts/kvmem_eval/paper_latency_agentlongbench_samples_20260812.jsonl`.
Every formal method is validated against the P50 identity from that manifest.
P25/P75 pilot measurements may be retained on disk but are excluded from final
aggregation.

For each slice:

1. Render each sample with the canonical worker used by its frozen utility run.
2. Tokenize the history without the final question using qw3 `/tokenize`.
3. Sort the frozen utility population by rendered history-token length.
4. Select the sample closest to P50, with stable sample ID as the deterministic
   tie-breaker.
5. Persist its ID and history/query token counts before launching a method.
6. Run every applicable method on exactly the same P50 ID.
7. Warm kernels only with an unrelated synthetic request, then restore the
   clean pre-query state and perform one recorded run.
8. If a run fails, retry the same sample from the clean state; do not replace
   it with another sample.
9. Report the one measured P50 value. Retain any earlier P25/P75 pilot values,
   but do not average them into the main table.

| Slice | Frozen population | Samples | Methods |
|---|---:|---:|---|
| AgentLongBench <=256K | 250 | 1 (P50) | Full Context, Sliding Window, Compact-only, Compact+RAG, KVMem |
| AgentLongBench 512K | 100 | 1 (P50) | Sliding Window, Compact-only, Compact+RAG, KVMem |
| AgentLongBench 1M | 50 | 1 (P50) | Sliding Window, Compact-only, Compact+RAG, KVMem |

This is 13 method-sample measurements:

```text
1 * 5 + 1 * 4 + 1 * 4 = 13
```

Compact-only and Compact+RAG may reuse the identical final boundary summary
artifact when their frozen compact configuration and input are identical. Both
main values include that one measured boundary-compaction cost; neither repeats
earlier maintenance rounds.

For AgentLongBench-1M, utility reuses the frozen question-blind DeepSeek V4 Pro
summary, but the historical July API timing is not used in the current latency
table. The P50 compaction call is replayed on the current host using qw3 and the
frozen prompt policy. Validation requires the original history identity and
server prompt-token count. The main Compact value charges this current model
time; the parenthesized value contains only current local RAG preparation
and/or final-query TTFT. Raw artifacts are under the formal run root.

```text
/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_current_20260812/
  alb1m_deepseek_compaction_current
```

The replay is launched by
`scripts/kvmem_eval/run_alb1m_deepseek_compaction_latency_sample.sh`.  It reads
the API key only from the process environment, copies it to a mode-0600 tmpfs
file for the upstream runner, and removes the file on exit.

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
- [x] Freeze P50 AgentLongBench IDs and rendered token lengths.
- [ ] Complete targeted MemoryAgentBench timing for the same 30 contexts/1,316 questions.
- [ ] Measure five methods on the AgentLongBench <=256K P50 sample.
- [ ] Measure four methods on the AgentLongBench-512K P50 sample.
- [ ] Measure four methods on the AgentLongBench-1M P50 sample.
- [ ] Verify that no Full Context run is launched above complete-history capacity.
- [ ] Verify every main KVMem cell uses client TTFT, with semantic/replay retained only as breakdown.
- [ ] Verify Compact main values include full non-amortized compaction and parentheses exclude it.
- [ ] Report `n=3` for AgentLongBench representative latency estimates.
