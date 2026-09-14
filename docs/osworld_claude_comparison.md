# Native Claude Code OSWorld comparison

The question is whether KVMem reduces real interruptions between desktop
observations and subsequent actions while preserving task performance. A fast
first summary token is not evidence of a short compaction interruption.

## Experimental units

The scored cohort uses the existing OSWorld-V2 task implementations and their
evaluators, with tasks `052, 004, 008, 053, 024, 103, 035, 066, 020, 049, 044,
010, 012, 013, 014`, seed 0, at most 500 desktop actions, and a 10,800-second
client execution budget per trial. The initial observation does not consume an
action. Setup and evaluation occur outside the client execution budget.

Each method starts from its own original task reset. Both use Claude Code
2.1.148, the same desktop MCP tool, screenshot processing, original task
instruction, and evaluator. Host tools and subagents are disabled. Each pair
shares an empty temporary agent working directory outside the source repository,
with Git discovery stopped above that directory. Source edits must not enter the
agent's system prompt as repository status. Method order alternates by task.

Ordinary qw3 retains Claude Code's native automatic compaction. Its native
compaction window is 262,144 tokens. KVMem disables client compaction and retains
the original full history. Neither method receives preloaded history or a custom
summary policy. The final cohort reruns 052 with the same isolated workspace
policy as the other fourteen tasks.

## Inference settings

Both methods use Qwen3.8-27B Q8_0 weights, FP8 KV, the same CUDA BF16 vision
encoder, MTP chain length 4, temperature 1, top-p 0.95, top-k 20, no thinking
budget, and the original 32,768-token per-request output limit. Frozen binary
hashes and complete server commands are recorded. Both methods use the same
validated CPU request-parser optimizations.

Ordinary qw3 uses a 262,144-token context and continuous batching with one active
request. KVMem uses a 2,097,152-token logical history, selection and prefill budgets
of 131,072 tokens, a 65,536-token generation reserve, 32-token blocks, a CPU
mean-k index, an 80 GiB CPU KV tier, and no NVMe tier. Guided query generation,
selection, query replay, and all GPU computations remain enabled. The guided
query limit remains 256 tokens; normal end-of-sequence termination is allowed.

CPU memory headroom and unrelated service state must remain the same throughout
both methods of a pair. Resource samples record process RSS/high-water mark,
swap, and host available memory. Temporarily stopped services must be restored
after testing. Image downloads do not use HTTP/HTTPS/ALL proxy variables.

## Measurements and interpretation

- **Task outcome:** retain the official score for every completed pair,
  including failures. Full success means score exactly 1; report mean official
  score separately if partial credit occurs. Time or action budget exhaustion
  ends a scored trial. An infrastructure or client execution error is preserved
  as an incomplete attempt requiring diagnosis.
- **Step TTFT:** previous desktop observation ready to the first content of the
  API response that produces the next desktop action. An intervening summary
  response remains inside this waiting interval.
- **Action wait:** previous observation ready to dispatch of the next desktop
  action, including response generation and client processing.
- **Maintenance:** report Claude Code's native compaction duration and the
  ensuing action wait. For KVMem, identify semantic-selection requests and
  separately report guided query, selection, query replay, and the ensuing
  action wait. Complete request TTFT is not the selection kernel's duration.
- **Ordinary steps:** report latency separately from maintenance steps and
  inspect its growth with history. Give equal weight to trials in the main
  per-trial mean and also expose the pooled per-step mean. Long unsuccessful
  trajectories must not silently determine the entire comparison.

`run_claude_suite.py` retains unfinished attempts, checks frozen configuration on
resume, and produces per-pair analysis. `summarize_claude_suite.py` lists pending
pairs explicitly and includes only complete pairs in comparative statistics.
It reports missing maintenance observations as missing, not as zero latency.
The single-seed cohort describes observed performance; it cannot establish
statistical equivalence of model capability.

OSWorld's `BaseTask.evaluate` supports both numeric returns and structured
returns whose `score` field contains the official numeric score. The summary
reads that score directly; auxiliary fields are retained and no penalties are
recomputed. Trial `result.json` remains unchanged. For structured returns, the
derived analysis result exposes numeric `score` and preserves the complete
payload as `official_evaluation`, including for older summary readers already
loaded by a running cohort. Task 008 first exercised this form in the current
cohort. Five score-format tests and six budget/maintenance tests verify the
compatibility change without altering the model, task or official evaluator.

Selection accounting covers both post-prefill selection and refreshes during
decode. A middecode refresh may begin after the first response content, so a low
TTFT does not prove the absence of a maintenance pause. The analyzer keeps
separate prefill/middecode request counts and the number of individual refreshes,
marks both kinds on the plots, and excludes both from ordinary-step statistics.
All model calls between an observation and the next action are considered,
including a non-action response before the final tool call. The per-action
middecode timer sums the recorded query and reselect phases once; the full action
wait also covers any final state rebasing outside those timers.

When auditing ordinary qw3 cache reuse, distinguish the long-history summary
request from the short context created after compaction. The former should reuse
the previous history prefix; the latter can legitimately need fresh KV for its
new summary. For example, in the archived diagnostic task 004, summary requests
98 and 202 reuse 228,528 and 227,536 tokens, with prefill times of 2.259 and
2.235 seconds. The following requests contain only 7,761 and 8,081 tokens and
build their new KV in 1.849 and 1.919 seconds. Their zero-reuse counters are not a
recurrence of the roughly 230K-token summary-request cache miss. Full action-wait
metrics include this post-compaction work.

## Desktop tool validation

The desktop executor validates arguments against the published MCP input schema
before any desktop action. In particular, `keys` must be an array of strings;
a string such as `[command, w]` is returned as a tool error, not expanded into
character-wise hotkeys or silently repaired. Valid commands and the advertised
schema are unchanged. Failed validation does not execute a desktop action or
consume an action step; the model receives the error and may correct its call.

The earlier `claudecode_matched15_stream_20260909` cohort is superseded because
its executor accepted malformed key strings. The audit found 37 such executed
calls in ordinary qw3 task 004 and two in KVMem task 052. Raw traces, official
scores and latency observations remain archived, but these trials are excluded
from the final capability comparison. Both arms and all tasks in the replacement
cohort use the same validated executor. The schema is unchanged, and all 656
schema-valid recorded commands are byte-identical before and after the fix.
Four focused desktop validation tests and six budget/maintenance tests pass.
The audit is recorded in
`benchmark/osworld_v2/claudecode_tool_validation_20260909/trajectory_validation_audit.json`.

## Fixed-input efficiency validation

The 500 recorded native requests in
`benchmark/osworld_v2/claudecode_fixed_trace_20260909` form an unscored replay.
Both processing paths receive identical complete request bytes, output limits,
sampling settings, and generated-query/replay computations. Returned tool calls
are recorded but never executed. Request inflation and output recording are
outside the measured API interval. Input hashes, normalized complete outputs,
token usage, stop reasons, selection schedules, and native phase accounting are
checked before attributing timing changes to the CPU optimization.

This replay can establish an efficiency change on fixed inputs. It cannot
replace a scored desktop trial or make differing live trajectories comparable
step for step. In particular, a live KVMem trial that ends before selection
does not measure KVMem maintenance cost.

For a subsequent CPU-only server build, the completed optimized replay can be
the control. The new replay must verify every original input hash, complete
output, usage and selection schedule again. Its provenance must identify the
unchanged inference and CUDA library, and explicitly list CPU configuration
differences. The new binary's own successful replay is required before expanding
the scored cohort; an older binary's validation is insufficient.

Request traces in the scored cohort use lossless grouped compression. The
original transmitted request bytes remain unchanged, and checksums cover their
complete restoration. Recording compression is unrelated to Claude Code
compaction or model-visible context management.
