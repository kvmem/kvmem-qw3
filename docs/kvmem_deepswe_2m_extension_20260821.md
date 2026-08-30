# DeepSWE KVMem 2M Extension

## Objective

After the existing ten-task ordinary QW3 control finishes naturally, run a
new, independent ten-task DeepSWE KVMem experiment with the logical context
increased from 1M to 2M. Keep the remaining frozen KVMem and Claude Code
parameters unchanged. Infrastructure-invalid samples may be resumed after a
fix; completed samples must not be rerun.

## Dependency gate

- Current control experiment:
  `benchmark/claude_deepswe_ab/recordings/ordinary256_auto_compact_requestplan10_seed73_20260820_f`
- Required gate: `complete_samples == total_samples == 10` and
  `experiment.done` exists.
- The new experiment must use a separate directory and must not mutate the
  current experiment manifest, frozen harness, state, or results.

The gate passed at 2026-08-22 00:58 CST. The ordinary control produced ten
valid natural completions out of ten samples and wrote `experiment.done` before
any image preparation, build, or 2M GPU work began.

## Frozen task selection

Catalog:
`benchmark/claude_deepswe_ab/tasks_stratified_extension_10_seed73.json`

The selection uses seed 73 and independent historical `peak_context_tokens`
from `benchmark/opencode_swebench/deepswe_v1_1_trials.json`. It excludes the
original RequestPlan-10 and does not use reward. Historical quartile allocation
is Q1/Q2/Q3/Q4 = 3/3/2/2.

## Intended configuration

Only the logical context changes relative to the completed
`kvmem128_gen64_requestplan10_seed73_20260820_e` experiment:

- logical context: 2,097,152 tokens (2M)
- selection/prefill budget: 131,072 tokens (128K)
- generation reserve: 65,536 tokens (64K)
- block size: 32 tokens
- retrieval: mean-k, GPU index
- immutable K: enabled
- query-conditioned and guided reselection: enabled
- guided query: 512 tokens
- mid-decode trigger: 61,440 tokens; maximum refreshes: 2
- KV dtype: FP8
- prefill chunk: 2,048
- MTP chain: 4, batched draft and paged prefix enabled
- sampling: temperature 1.0, top-p 0.95, top-k 20, seed 73
- thinking guard: 28,672 tokens
- Claude Code compaction and subagents: disabled
- concurrency: one; fresh QW3 service per task

CPU backing capacity is a resource capacity, not a model-policy parameter. The
1M run reserved 40 GiB (about 17 GiB immutable raw-K authority plus 23 GiB
spill headroom), so the 2M run preserves the same per-token capacity with an
80-GiB CPU tier. This does not change the active budget, retrieval, sampling or
KVMem semantics. The value is frozen in the new experiment manifest.

## Resume and validity rules

1. Never rerun an arm that already has valid `metrics.json` and a finished
   status.
2. A reward of zero is a valid model result, not a reason to rerun.
3. Pause on HTTP/stream errors, CUDA failures, abnormal Claude termination,
   subagent use, dependency/network preflight failures, or verifier failure.
4. Diagnose and fix the fault, preserve the failed attempt under `attempts/`,
   then resume from the affected sample only.
5. Keep the task list, ordering, sampling and model frozen after the first arm
   starts.

## Issue log

| ID | Time | Sample | Symptom | Classification | Resolution | Verification |
|---|---|---|---|---|---|---|
| PRE-001 | 2026-08-21 | preflight | Nine of ten selected Agent base images and their verifier images are not installed; only `meriyah-explicit-resource-declarations` is present | Environment prerequisite | After the ordinary 10/10 completion gate passed, pulled the nine pinned Agent images and built their verifier images before initializing the 2M experiment | All ten Agent and verifier image pairs reported `ready`; the frozen manifest was initialized only after the image-preparation script completed |
| PRE-002 | 2026-08-21 | preflight | The KVMem-only harness and server condition hard-code a 1,048,576-token logical context in the manifest, Claude environment, readiness gate and QW3 `--ctx` | Harness limitation | Added new-experiment-only `--kvmem-context`; the manifest, Claude environment, QW3 server environment and readiness markers now consume the same value. Existing manifests retain their frozen value and the default remains 1M | Static checks passed, then the first live service reported `ctx=2097152`, an 80-GiB CPU tier and Claude model id `Qwen3.8-27B[2m]`; its first request completed successfully |
| RUN-001 | 2026-08-22 | sample 8, `goreleaser-retry-publish-auditing` | Dependency preflight exited before Claude started: the image's bootstrap Go 1.25.5 rejected `go.mod`'s Go 1.26.1 requirement, even though the pinned image already contained the complete cached Go 1.26.1 toolchain | Harness bug; invalid sample attempt | Replaced the runner's forced `GOTOOLCHAIN=local` with `GOTOOLCHAIN=auto` while retaining the task's network-isolated container and `GOPROXY=off`; the failed arm is preserved under `attempts/` and only sample 8 is rerun | The exact pinned image passed `go version` and `go list -mod=readonly -deps ./...` as `nobody` with the cached Go 1.26.1 toolchain and no network; the 31-test harness suite passed. The live rerun recorded `gotoolchain=auto`, `status=passed`, completed both no-WAN checks, and started Claude normally. |
| RUN-002 | 2026-08-22 | sample 8, `goreleaser-retry-publish-auditing` | Claude invoked `WebSearch` twice in a task whose container is intentionally network-isolated; the local Anthropic adapter returned two standalone HTTP 400 responses for that unsupported external-tool path | Expected offline tool limitation; recoverable Agent tool outcome, not a KVMem inference/checkpoint failure | No serving change and no rerun. Keep the events visible; future no-network harnesses may hide `WebSearch` or return an explicit deterministic unavailable result | Both attempts returned to the Agent, which continued naturally. All 176 accounted model-inference requests completed with HTTP 200, `stream_completed=1`, and `terminal_status=success`; Claude reported terminal success, agent/verifier exits were 0/0, and no checkpoint, recovery, MTP, CUDA, OOM, or stream-terminal fault occurred |

## Live run status

The frozen experiment was initialized at 2026-08-22 01:06 CST under:

`benchmark/claude_deepswe_ab/recordings/kvmem2m_128gen64_stratified10_seed73_20260821_a`

The first task, `kcp-go-multiplexed-kcp-streams`, started after a clean build and
readiness gate. Comparing the effective serving-parameter blocks against the
completed 1M reference showed that all KVMem, MTP and sampling parameters are
identical. The intended capacity-only differences are logical context 1M to 2M
and CPU backing 40 GiB to 80 GiB; the container bridge address also changed.
The removed experimental vision bridge no longer appears, which is unrelated
to these text-only tasks.

The first task also exercised the decode-time threshold-crossing case in a
natural Claude Code trajectory. Request 21 began at 127,949 prompt tokens,
below the 131,072-token selection budget, and generated 6,982 tokens, leaving a
134,931-token warm state above the budget. Its capture retained
`query_snapshot=1`, `P_source_index=1`, and `M_source_index=1`. The following
tool-result continuation hit the M checkpoint at 134,931 and prefetched only a
43-token suffix; it did not report `no_reusable_checkpoint` or rebuild the
134K history. Because that continuation contained no new user query, the
request plan correctly used `keep-selected-append` rather than performing an
unsolicited semantic reselection.

The same trajectory subsequently exercised the first private-query refresh at
about 192.6K logical tokens. QW3 generated a 250-token retrieval query and
selected 4,096 of 6,019 source blocks. Scoring took 444 ms, materialization
93 ms, and the complete semantic reselection took 547 ms; private-query
generation itself took 6.47 s. The MTP path reported `mtp_rebase=bridge`, the
post-refresh capture had `P_rebased_after_selection=1`, and the request ended
with `stream_completed=1` and `terminal_status=success`. The following
continuation reused the post-selection P checkpoint and replayed/appended 919
tokens. No plain-MTP fallback, unprimed MTP V page, checkpoint-hole recovery,
CUDA error, or HTTP/stream failure was observed.

A second natural private-query refresh at about 254.1K logical tokens also
completed successfully. It selected 4,096 of 7,940 source blocks; scoring took
469 ms and the complete reselection took 512 ms. The CPU stage-in path supplied
146 blocks (0.076 GiB), MTP again reported `mtp_rebase=bridge`, and execution
continued with selection generation 23. At this checkpoint the complete server
log still contained zero occurrences of `unexpected_no_reusable_checkpoint`,
`mtp_fallback=plain`, unprimed MTP V errors, `recovery-reselect`, CUDA/OOM, or
terminal stream errors.

The second task, `ofetch-per-origin-circuit-breaker`, independently exercised
the decode-time selection-budget crossing. Request 32 began with a 122,360-token
prompt and generated 9,123 tokens; the following request therefore arrived at
131,529 prompt tokens, above the 131,072-token selection budget. It restored the
M checkpoint at 131,483 with `query_snapshot=1` and appended only a 46-token
suffix. Because the continuation contained neither a new user query nor a
private-query trigger, the request correctly used `keep-selected-append`
instead of performing a semantic reselection with the stale task query. It did
not report `no_reusable_checkpoint`, `recovery-reselect`, an MTP fallback, or a
stream error. This natural crossing provides a second confirmation that
below-budget preparation prevents the former first-over-budget cold rebuild.

The same task later exercised a real guided mid-decode refresh at logical
position 192,580. The generated private query contained 476 tokens and selected
4,096 of 6,019 source blocks. Of these, 2,588 blocks remained reusable and
1,508 entered the new active selection. Scoring took 833.3 ms, materialization
86.5 ms, and the complete reselection took 929.3 ms. Private-query generation
took 12.31 s, while private-prompt construction and recurrent-state handling
took 106.6 ms and 4.5 ms respectively. MTP reported `mtp_rebase=bridge`. The
post-refresh checkpoint capture had `P_rebased_after_selection=1`, selection
generation 33, ready P/M source indexes, and `query_snapshot=1`. Request 60
then completed successfully, and request 61 restored the M checkpoint at
194,766 and processed only a 615-token suffix. No checkpoint-hole recovery,
plain-MTP fallback, unprimed MTP V error, CUDA error, or terminal stream error
occurred.

Its second guided refresh occurred at logical position 254,033. The generated
private query contained 82 tokens and selected 4,096 of 7,939 source blocks.
Scoring took 199.9 ms, materialization took 29.9 ms, and the complete semantic
reselection took 255.1 ms. The CPU tier supplied 112 blocks (0.058 GiB), private
query generation took 2.18 s, query replay took 715 ms, and MTP again used the
bridge rebase path. The request completed with `stream_completed=1` and the
remaining seven requests all resumed a checkpoint successfully.

The third task, `igel-persist-feature-schema`, supplied a particularly tight
selection-budget crossing. Request 96 started at 130,886 prompt tokens and
ended with a 131,028-token warm state, only 44 tokens below the 131,072-token
budget. Request 97 arrived at 131,543 prompt tokens. It restored the M
checkpoint at 131,028 with `query_snapshot=1`, ready P/M source indexes, and
prefilled only the 515-token suffix. With no new user query or private-query
trigger, the request plan correctly changed from `dense-append` to
`keep-selected-append` without performing a semantic reselection. The request
completed successfully, and the following requests continued to resume their
M checkpoints. There was no cold rebuild, recovery reselection, MTP fallback,
unprimed MTP V error, CUDA error, or terminal stream error.

The fourth task, `koota-entity-snapshot-rollback`, crossed the selection budget
through ordinary incremental continuations and then exercised its first
private-query refresh at 193,790 logical tokens. The request restored the P
checkpoint at 190,912, replayed/appended a 2,878-token suffix, and generated a
280-token private retrieval query in 7.28 seconds. Mean-K selected 4,096 of
6,056 source blocks. Scoring took 502.2 ms, materialization took 87.1 ms, and
the complete semantic reselection took 607.2 ms. The request completed with
`stream_completed=1`; its new capture had `P_rebased_after_selection=1`,
selection generation 55, ready P/M source indexes, and `query_snapshot=1`.
The following continuation resumed the new checkpoint. At this point the
service still had zero checkpoint-hole recoveries, unexpected no-reusable
checkpoints, plain-MTP fallbacks, unprimed MTP V errors, CUDA errors, HTTP
4xx/5xx responses, or terminal stream errors.

The fifth task, `kombu-single-active-consumer-priority`, supplied an especially
large natural selection-budget crossing. A request began at 126,115 prompt
tokens and generated 16,834 tokens, producing a 142,949-token warm state in a
single decode. The next request arrived at 142,999 tokens, restored the complete
M checkpoint at 142,949 with its query snapshot and source indexes, and
processed only the 50-token suffix. The plan correctly changed from
`dense-append` to `keep-selected-append`; it neither rebuilt the 143K history
nor performed a semantic reselection with the stale task query. The following
continuation again restored its M checkpoint. No checkpoint-hole recovery,
plain-MTP fallback, unprimed MTP V error, CUDA error, or HTTP/stream failure was
observed through this crossing.

The same fifth trajectory later completed its first guided mid-decode refresh
at logical position 192,528. It generated a 343-token private retrieval query
in 8.86 seconds, selected 4,096 of 6,017 source blocks, and kept 16,384 recent
tokens. Mean-K scoring took 608.8 ms, materialization took 93.3 ms, and the
complete semantic reselection took 712.5 ms. MTP reported
`mtp_rebase=bridge`; the post-refresh capture had
`P_rebased_after_selection=1`, ready P/M source indexes, and
`query_snapshot=1`. All subsequent continuations through at least 212K logical
tokens resumed their M checkpoint and processed only the new suffix. The live
service log contained 105 MTP checkpoint hits and 105 captures at the audit
point, with zero occurrences of `unexpected_no_reusable_checkpoint`,
`no_reusable_checkpoint`, `recovery-reselect`, `mtp_fallback=plain`, unprimed
MTP V errors, CUDA errors, incomplete streams, or terminal request errors.

Its second guided refresh occurred at logical position 253,985. The private
query generation reached the configured 512-token cap and therefore used the
documented `fallback-original` query mode; this is a bounded query-generation
outcome rather than a serving failure. Mean-K selected 4,096 of 7,938 source
blocks, retaining 3,041 blocks and admitting 1,055 incoming blocks. Scoring
took 794.7 ms, assembly took 23.7 ms, and the complete semantic reselection
took 829.0 ms. No CPU or NVMe stage-in was exposed because the incoming blocks
were already available in the active GPU pool. MTP again used the bridge rebase
path, and the post-refresh capture had `P_rebased_after_selection=1`, ready P/M
source indexes, and `query_snapshot=1`. The request completed normally and the
trajectory continued beyond 257K logical tokens with all serving-fault
counters still at zero.

The sixth task, `bandit-structured-nosec-directives`, independently exercised
another large decode-time selection-budget crossing. Request 28 began at
121,542 prompt tokens and generated 18,018 tokens, producing a 139,560-token
warm state above the 131,072-token budget. Its capture preserved
`query_snapshot=1`, ready P/M source indexes, and a resumable M checkpoint.
The next continuation arrived at 139,665 tokens, changed from `dense-append`
to `keep-selected-append`, restored M at 139,560, and processed only the
105-token suffix. It did not cold-prefill the 140K history or perform a stale
query semantic reselection. The next two requests again resumed M with small
suffixes. No unexpected missing checkpoint, recovery reselection, plain-MTP
fallback, unprimed MTP V error, CUDA error, incomplete stream, or terminal
request error occurred through this crossing.

At 192,718 logical prompt tokens, the sixth task reached the initial-headroom
request-boundary trigger and correctly changed its plan to
`generate-private-query-then-reselect`. QW3 generated a 300-token private
retrieval query in 7.83 seconds and selected 4,096 of 6,023 source blocks.
Mean-K scoring took 533.6 ms, materialization took 95.4 ms, and the complete
semantic reselection took 650.9 ms. Query replay rolled back to the 191,584
boundary and replayed a 1,134-token suffix against the fixed selected context.
The post-selection MTP capture had `P_rebased_after_selection=1`, ready P/M
source indexes, and `query_snapshot=1`. The next request at 193,247 restored
the M checkpoint at 193,151 and processed only a 96-token suffix. The private
query remained internal, and all serving-fault counters remained zero.

The seventh task, `cliffy-config-file-parsing`, crossed the selection budget
when a request at 128,206 prompt tokens generated 3,917 tokens and captured a
132,123-token warm state with `query_snapshot=1`, ready P/M source indexes, and
both checkpoints marked resumable. The following prompt contained 132,181
tokens but its exact LCP ended at 132,118, five tokens before the M checkpoint,
so the safety gate correctly selected the earlier P checkpoint at 128,160 and
replayed/appended 4,021 tokens. It did not trust the five-token-incompatible M
state, cold-prefill the full 132K history, or perform a stale-query semantic
reselection. This is a safe P-tail replay caused by a small prompt-tail rewrite,
not a missing-checkpoint failure. All serving-fault counters remained zero.

The same seventh trajectory then triggered its first guided mid-decode refresh
at logical position 192,538, after 850 decode tokens in the request. It
generated a 124-token private retrieval query in 3.21 seconds and selected
4,096 of 6,017 source blocks while preserving 16,384 recent tokens. Mean-K
scoring took 226.3 ms, materialization took 95.1 ms, and the complete semantic
reselection took 331.7 ms. MTP reported `mtp_rebase=bridge`; the post-refresh
capture had `P_rebased_after_selection=1`, ready P/M source indexes, and
`query_snapshot=1`. The request completed successfully and the next request at
194,423 restored the M checkpoint at 192,711 before processing the new suffix.
The private query remained internal and every serving-fault counter remained
zero.

The eighth task, `goreleaser-retry-publish-auditing`, passed the repaired
offline Go dependency preflight and later reached its first genuine semantic
reselection at 193,614 logical prompt tokens. The request restored the P
checkpoint at 189,952 and replayed only the 3,662-token suffix rather than
rebuilding the full history. Mean-K selected 4,096 of 6,051 source blocks,
retaining 2,545 blocks and admitting 1,551 incoming blocks. Scoring took
167.0 ms, materialization took 93.8 ms, and the complete semantic reselection
took 289.0 ms. Query replay forward execution took 1.81 seconds. The
post-selection capture had `P_rebased_after_selection=1`, selection generation
70, ready P/M source indexes, `query_snapshot=1`, and a resumable M checkpoint.
The next request restored that M checkpoint and all subsequent continuations
through the final 241,605-token prompt retained selection generation 70. This
trajectory did not emit the literal
`mtp_rebase=bridge` diagnostic at the first selection, but post-selection MTP
continuity is directly established by the resumable P/M capture and the next
M-checkpoint hit. No unexpected/no-reusable checkpoint, recovery reselection,
plain-MTP fallback, unprimed MTP V error, CUDA error, incomplete stream, or
terminal request error occurred in the completed trajectory.

The ninth task, `katex-multicolumn-array-spans`, independently exercised a
request-boundary selection-budget crossing. A request at 130,375 prompt tokens
generated 128 tokens and captured a 130,503-token warm state with
`query_snapshot=1`, ready P/M source indexes, and a resumable M checkpoint.
The following request arrived at 131,087 prompt tokens, just above the
131,072-token budget. It changed from `dense-append` to
`keep-selected-append`, restored M at 130,503, and processed only the
584-token suffix. Its new 131,474-token capture retained the same complete
checkpoint metadata. The next request at 132,407 again restored M at 131,474
and processed only its 933-token suffix. No cold history rebuild, stale-query
semantic reselection, unexpected/no-reusable checkpoint, recovery reselection,
plain-MTP fallback, unprimed MTP V error, CUDA error, incomplete stream, or
terminal request error occurred through this crossing.

The same ninth trajectory reached its first guided refresh at 192,554 logical
prompt tokens. It generated a 142-token private retrieval query in 3.76
seconds and selected 4,096 of 6,018 source blocks. The new selection retained
2,558 blocks and admitted 1,538 incoming blocks. Mean-K scoring took 259.2 ms,
materialization took 96.2 ms, and the complete semantic reselection took
374.7 ms. Query replay restored the P boundary at 191,648 and replayed only the
906-token suffix in 499.4 ms. The post-selection capture had
`P_rebased_after_selection=1`, selection generation 40, ready P/M source
indexes, `query_snapshot=1`, and a resumable M checkpoint. The next request at
193,031 restored that M checkpoint and processed only a 45-token suffix. The
private query remained internal and every serving-fault counter remained zero.

Its second guided refresh occurred at 254,605 logical prompt tokens after
62,051 tokens of cross-turn growth. It generated a 120-token private query in
3.19 seconds and selected 4,096 of 7,957 source blocks. The new selection
retained 2,827 blocks and admitted 1,269 incoming blocks. Mean-K scoring took
278.0 ms, materialization took 31.1 ms, and the complete semantic reselection
took 339.4 ms. The CPU tier supplied 71 blocks (0.037 GiB) in a 6.6-ms exposed
stage-in, with no NVMe access. Query replay restored the P boundary at 246,816
and replayed the 7,789-token suffix in 3.73 seconds. The post-selection capture
had `P_rebased_after_selection=1`, selection generation 41, ready P/M source
indexes, `query_snapshot=1`, and a resumable M checkpoint. The request
completed with a successful stream, the private query remained internal, and
all serving-fault counters remained zero.

The ninth trajectory later issued a third guided refresh at logical position
316,100. This is expected: `middecode_max_refreshes=2` limits refreshes within
one API request, not across the complete multi-request Agent trajectory. The
refresh occurred in a later request after another cross-turn growth interval,
generated a 390-token private retrieval query, and selected 4,096 of 9,879
source blocks. The new selection retained 2,537 blocks and admitted 1,559
incoming blocks. Mean-K scoring took 226.5 ms, materialization took 38.1 ms,
and the complete semantic reselection took 276.7 ms. The post-refresh capture
had `P_rebased_after_selection=1`, selection generation 42, ready P/M source
indexes, `query_snapshot=1`, and a resumable M checkpoint. The trajectory then
completed naturally at a maximum 363,770-token prompt. All three private
queries remained internal, all 208 continuation requests resumed a prefix
checkpoint, and no serving fault occurred.

The tenth task, `meriyah-explicit-resource-declarations`, also crossed the
131,072-token selection budget without a cold rebuild. Its final below-budget
request arrived with 127,172 prompt tokens, restored the M checkpoint at
125,914, processed only the 1,258-token suffix, generated 2,106 tokens, and
captured a 129,278-token warm state with `query_snapshot=1`, ready P/M source
indexes, and a resumable M checkpoint. The next request arrived with 131,319
prompt tokens. The request plan changed from `dense-append` to
`keep-selected-append`, restored M at 129,278, and processed only the
2,041-token suffix. Its 132,423-token capture retained the complete checkpoint
metadata and did not perform a semantic or recovery reselection. The following
request at 133,787 again restored M at 132,423 and processed only its
1,364-token suffix. No unexpected/no-reusable checkpoint, stale-query recovery
reselection, plain-MTP fallback, unprimed MTP V error, CUDA error, incomplete
stream, or terminal request error occurred through this crossing.

The tenth trajectory reached its first guided refresh when a new request
arrived with 194,106 prompt tokens. The request plan reported
`trigger=initial-headroom`, generated a 108-token private retrieval query in
2.87 seconds, restored the safe P checkpoint at 183,872, and replayed the
10,234-token suffix rather than rebuilding the complete history. Mean-K
selected 4,096 of 6,066 source blocks, retaining 2,359 blocks and admitting
1,737 incoming blocks. Scoring took 202.2 ms, materialization took 91.8 ms,
and the complete semantic reselection took 326.0 ms. The resulting 197,174
capture had `P_rebased_after_selection=1`, selection generation 54, ready P/M
source indexes, `query_snapshot=1`, and a resumable M checkpoint. The next
request restored that state and all serving-fault counters remained zero.

## Completed samples

| # | Task | Agent / verifier | Reward | F2P | P2P | Max logical prompt | Actual semantic reselections | Serving validity |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `kcp-go-multiplexed-kcp-streams` | 0 / 0 | 0 | 2/30 | 12/12 | 288,162 | 2 | Valid natural completion; 120/120 continuation requests resumed a checkpoint, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |
| 2 | `ofetch-per-origin-circuit-breaker` | 0 / 0 | 1 | 47/47 | 13/13 | 262,456 | 2 | Valid natural completion; 116/116 continuation requests resumed a checkpoint, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |
| 3 | `igel-persist-feature-schema` | 0 / 0 | 0 | 6/24 | 2/2 | 149,999 | 0 | Valid natural completion; 119/119 continuation requests resumed a checkpoint, including the 128K crossing, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |
| 4 | `koota-entity-snapshot-rollback` | 0 / 0 | 1 | 84/84 | 47/47 | 204,898 | 1 | Valid natural completion; 114/114 continuation requests resumed a checkpoint, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |
| 5 | `kombu-single-active-consumer-priority` | 0 / 0 | 1 | 85/85 | 1421/1421 | 261,976 | 2 | Valid natural completion; 139/139 continuation requests resumed a checkpoint, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |
| 6 | `bandit-structured-nosec-directives` | 0 / 0 | 0 | 69/69 | 281/282 | 228,875 | 1 | Valid natural completion; 91/91 continuation requests resumed a checkpoint, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |
| 7 | `cliffy-config-file-parsing` | 0 / 0 | 0 | 36/37 | 451/451 | 225,605 | 1 | Valid natural completion; 151/151 continuation requests resumed a checkpoint, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |
| 8 | `goreleaser-retry-publish-auditing` | 0 / 0 | 0 | 7/29 | 29/29 | 241,605 | 1 | Valid natural completion after RUN-001; 175/175 continuations resumed a checkpoint and all 176 inference streams succeeded. Two separately documented offline `WebSearch` tool calls returned HTTP 400 (RUN-002); no KVMem/checkpoint/MTP/CUDA/stream-terminal fault occurred |
| 9 | `katex-multicolumn-array-spans` | 0 / 0 | 0 | 92/94 | 599/599 | 363,770 | 3 | Valid natural completion; 208/208 continuation requests resumed a checkpoint, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |
| 10 | `meriyah-explicit-resource-declarations` | 0 / 0 | 1 | 49/49 | 51469/51469 | 232,682 | 1 | Valid natural completion; 132/132 continuation requests resumed a checkpoint, including the 128K crossing and guided refresh, with zero recovery, checkpoint-hole, MTP fallback/unprimed-V, HTTP 4xx/5xx, or stream-terminal errors |

Sample 1 ran for 4,341 seconds in the Agent and 4,552 seconds including the
verifier. It made 121 model requests and 151 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. Its reward of zero is an accepted model result and is not a reason
to rerun it. The durable queue stopped the first fresh QW3 service and advanced
to sample 2 without mutating the completed arm.

Sample 2 ran for 2,778 seconds in the Agent and 2,786 seconds including the
verifier. It made 117 model requests and 144 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. The verifier passed all 47 F2P and 13 P2P tests for reward 1. The
durable queue preserved both completed arms, stopped the second fresh service,
and advanced to sample 3, `igel-persist-feature-schema`.

Sample 3 ran for 1,401 seconds in the Agent and 1,414 seconds including the
verifier. It made 120 model requests and 139 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. It ended at 149,999 prompt tokens before a private-query refresh was
needed. The verifier passed 6/24 F2P and 2/2 P2P tests for reward 0; this is an
accepted model result. The queue preserved all three completed arms and
advanced to sample 4, `koota-entity-snapshot-rollback`.

Sample 4 ran for 2,162 seconds in the Agent and 2,173 seconds including the
verifier. It made 115 model requests and 156 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. The verifier passed all 84 F2P and 47 P2P tests for reward 1. Its
single private retrieval query remained private, all 114 continuation requests
resumed a checkpoint, and every serving-fault counter remained zero. The
durable queue preserved all four completed arms, stopped the fourth fresh QW3
service, and advanced to sample 5, `kombu-single-active-consumer-priority`.

Sample 5 ran for 2,447 seconds in the Agent and 2,492 seconds including the
verifier. It made 140 model requests and 154 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. The verifier passed all 85 F2P and all 1,421 P2P tests for reward 1.
Both private retrieval queries remained private, all 139 continuation requests
resumed a checkpoint, and every serving-fault counter remained zero. Its two
recorded tool errors were ordinary Agent tool outcomes rather than API,
serving, CUDA, or verifier failures. The durable queue preserved all five
completed arms, stopped the fifth fresh service, and advanced to sample 6,
`bandit-structured-nosec-directives`.

Sample 6 ran for 2,461 seconds in the Agent and 2,487 seconds including the
verifier. It made 92 model requests and 131 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. The verifier passed all 69 F2P tests and 281/282 P2P tests, so the
binary reward is zero while the partial verifier score is 0.99715. This is a
valid model result, not an infrastructure failure. Its single private query
remained internal, all 91 continuation requests resumed a checkpoint, and
every serving-fault counter remained zero. The durable queue preserved all six
completed arms, stopped the sixth fresh service, and advanced to sample 7,
`cliffy-config-file-parsing`.

Sample 7 ran for 1,902 seconds in the Agent and 1,922 seconds including the
verifier. It made 152 model requests and 159 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. The verifier passed 36/37 F2P tests and all 451 P2P tests, so the
binary reward is zero while the partial verifier score is 0.99795. This is a
valid model result, not an infrastructure failure. Its single private query
remained internal, all 151 continuation requests resumed a checkpoint, and
every serving-fault counter remained zero. The durable queue preserved all
seven completed arms, stopped the seventh fresh service, and advanced to
sample 8, `goreleaser-retry-publish-auditing`.

Sample 8 ran for 2,375 seconds in the Agent and 2,581 seconds including the
verifier. It made 176 model requests and 217 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. The verifier passed 7/29 F2P tests and all 29 P2P tests, so the
binary reward is zero while the partial verifier score is 0.62069. This is a
valid model result, not an infrastructure failure. Its single private query
remained internal, all 175 continuation requests resumed a checkpoint, and
every KVMem/checkpoint/MTP/CUDA/stream-terminal counter remained zero. Three of
its five recorded tool errors were ordinary Agent tool outcomes; the other two
were the explicitly documented offline `WebSearch` HTTP 400 results in
RUN-002. Neither interrupted a model-inference stream. The durable queue
preserved the valid rerun and its earlier
preflight-only failed attempt separately, stopped the eighth fresh service,
and advanced to sample 9, `katex-multicolumn-array-spans`.

Sample 9 ran for 3,556 seconds in the Agent and 3,569 seconds including the
verifier. It made 209 model requests and 247 completed tool calls, used no
subagent and no compaction, and finished with Claude terminal subtype
`success`. The verifier passed 92/94 F2P tests and all 599 P2P tests, so the
binary reward is zero while the partial verifier score is 0.99711. This is a
valid model result, not an infrastructure failure. Its three private queries
remained internal, all 208 continuation requests resumed a checkpoint, and
every serving-fault counter remained zero. The durable queue preserved the
ninth completed arm, stopped its fresh service, and advanced to sample 10,
`meriyah-explicit-resource-declarations`.

Sample 10 ran for 1,933 seconds in the Agent and 2,002 seconds including the
verifier. It made 133 model requests and completed naturally with no subagent
and no compaction. The verifier passed all 49 F2P tests and all 51,469 P2P
tests for reward 1. Its private query remained internal, all 132 continuation
requests resumed a checkpoint, and every serving-fault counter remained zero.
The durable supervisor stopped the tenth fresh service and exited with return
code 0.

## Final result

The experiment completed all 10/10 samples naturally and protocol-cleanly.
Agent and verifier exits were 0/0 for every sample. Binary reward was 4/10 and
the mean partial verifier score was 0.82539. Across the ten trajectories, the
server completed 1,375 accounted inference requests and 14 semantic guided
reselections; the maximum logical prompt was 363,770 tokens. Every accounted
inference request completed successfully, all 1,365 continuation requests
resumed a prefix checkpoint, and private-query leak count was zero. No
unexpected/no-reusable checkpoint, stale-query recovery reselection,
plain-MTP fallback, unprimed MTP V, CUDA, OOM, incomplete inference stream, or
terminal inference error occurred. RUN-001 is the only invalid attempt and is
preserved separately; RUN-002 records two recoverable offline `WebSearch` tool
errors without reclassifying the valid completed sample.

## Durable launch

`benchmark/claude_deepswe_ab/run_kvmem2m_extension_after_ordinary.sh` waits for
the ordinary completion artifact and validates 10/10 natural completions before
doing any image, build or GPU work. It then prepares the frozen images,
initializes the 2M manifest and runs the existing completed-arm-granular durable
supervisor. A model reward of zero remains complete; an explicit investigation
pause is not automatically restarted.
