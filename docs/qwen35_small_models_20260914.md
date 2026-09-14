# Qwen3.5 9B / 4B adaptation and validation

Baseline commit: `726d149`. User authorized adaptation and tests on 2026-09-14.

## Acceptance

* Qwen3.5-9B and 4B load from Unsloth Q8_0 GGUFs with verified upstream SHA-256 and pinned revisions. Download via hf-mirror with HTTP_PROXY, HTTPS_PROXY and ALL_PROXY removed in both cases.
* Preserve FP8 KV, MTP chain 4, full model computation, KVMem mean-K selection, guided query and replay. Ordinary inference keeps native Claude Code compaction.
* Validate HF nested RoPE configuration and dynamic MTP layer mapping; instantiate GQA=4 BatchDecode without changing the GQA=6 path.
* Parameterize the GPU vision frontend for each model's own tower and output width; check numerical parity for the 64-dimensional vision heads and retain 27B parity.
* Verify short generation, MTP on/off greedy consistency, paged continuous batching, prefix reuse and selection/replay beyond the configured budget. Include a 27B regression.
* Run a fixed DeepSWE pilot: the first eight IDs of the existing 16-task list, seed=1000, both context policies for each smaller model. Do not select tasks based on pilot scores. Retain failures and raw traces, report accuracy and latency separately.

## Resource and configuration policy

Use a separate build and result directory. Existing OSWorld uses its frozen binary and files. Download and CPU build can proceed while OSWorld runs. GPU timing and complete agent trials require exclusive GPU ownership; pause at an arm boundary and resume the original suite afterward, preserving completed results.

Before pilot launch, record exact commands and environment, model hashes, task IDs, Claude Code version, and the matching 27B configuration. The current OSWorld reference is Q8_0 / FP8 / MTP=4, plain ctx=262144 and KVMem ctx=2097152, selection=131072, generation=131072, block=32, CPU history/index, NVMe=0, GC off. Historical DeepSWE has a different generation budget (65536); do not silently pool those measurements with the new pilot or claim a matched cross-model comparison if the budget differs.

## Evidence location

`benchmark/qwen35_small_20260914/` holds manifests, download status, model binding checks, numerical checks, throughput measurements and the pilot's raw traces. Only validated results are reported as completed. Static compatibility and successful compilation alone do not satisfy the goal.

## Qualification and pilot status (2026-09-14)

The text build is frozen at SHA-256
`d6cf84c7a3e244ce1667e83b8b6034f3506a78784d92c6f862721ad75fbcb4e3`.
21 CPU tests and six independent attention-oracle cases passed (GQA 4/6,
FP8/FP16). Both models completed 8K/128K generation, MTP, prefix cold/warm
restoration and actual semantic selection/query replay. Guided generation was
also exercised. The 27B text and vision regression passed. Free-running MTP
on/off outputs are not exactly equal under FP8 batch arithmetic; they must not
be described as an exact greedy-equivalence result.

The Unsloth main Q8_0 files do not include MTP. Official MTP tensors were
downloaded separately, converted with the recorded llama.cpp converter and
appended; every main-model Q8 tensor was checked byte for byte. Auxiliary
`*-BF16` directories contain only vision/MTP tensors, not complete BF16 language
weights. Download provenance records repository revisions, sizes and hashes.

Small-model GPU vision is not yet qualified: the existing cosine >0.999 gate
gave 0.998719 (4B) and 0.998071 (9B) on the selected screenshot, with small
initial rounding differences amplified across layers. This limitation does
not apply to the text-only pilot. Do not report these as passed vision tests
or start a small-model OSWorld experiment without resolving this gate.

The pilot copies the existing 27B DeepSWE profile: Q8_0, FP8 KV, MTP=4,
selection 131072, generation 65536, block 32, logical history 3145728,
CPU backing cap 110 GiB, guided query limit 4096, thinking cap 28672,
temperature 1.0/top-p 0.95/top-k 20. Mean-K index placement remains GPU,
as in that DeepSWE reference. Ordinary context and native auto-compaction
window are 262144. NVMe and GC are not enabled. The fixed first eight tasks,
seed 1000, two models and two policies give 32 trials. After the first 4B pair,
the user requested 9B first: all sixteen 9B arms now precede the remaining 4B
arms, retaining the original within-model counterbalanced policy order.
`pilot_plan.json` records this schedule revision and the original plan/source
are archived under `controller_revision_9b_first/`. Model binaries, parameters,
task membership and completed results are unchanged. Existing agent scripts, dependency/network
preflight and verifier images are reused. Infrastructure errors pause the
queue and task failures remain in the data.

Run `python3 scripts/run_qwen35_deepswe_pilot.py --prepare-only` to audit the
manifest, and `--run` to start/resume. `pilot_plan.json`, per-arm manifests and
`pilot_status.json` record the frozen artifacts and progress. OSWorld was
paused after 060 completed; all 50 completed arm-result hashes are recorded
in `pause.json` and must remain unchanged when resuming.

The first completed 4B pair (`superjson-error-stack-serialization`) had reward
0 in both arms: 0/80 feature tests and 116/116 existing tests passed. Both
agents and verifiers exited normally; retain these quality failures. Weighted
decode throughput was 128.74 token/s for ordinary qw3 and 125.53 token/s for
KVMem, over different trajectories (82,586 vs 101,100 decoded tokens). KVMem
performed one real guided refresh: 919.6 ms query generation, 141.5 ms
selection/KV organization, 301.5 ms replay and 2.027 s visible-content TTFT.
No aggregate ability or speedup claim is supported by this one pair.

An additional visual diagnostic (`vision_backend_audit/comparison.json`)
compared native BF16 output and PyTorch math/Flash BF16 output against a
PyTorch FP32 math reference, without changing the original pass threshold.
For 4B, relative RMS error against FP32 was 8.57% native, 9.23% PyTorch math
BF16 and 8.76% PyTorch Flash BF16; for 9B, 9.35%, 9.27% and 9.28% respectively.
Even the two PyTorch BF16 attention implementations differed at cosine
0.998646 (4B) and 0.998455 (9B). This is evidence of numerical sensitivity,
not proof of full vision compatibility: the original native-vs-reference
gate is still failing and needs separate resolution before small-model
multimodal experiments.

Audit correction: that FP32 reference retained FP32 image preprocessing,
whereas the native and both BF16 references rounded input patches to BF16.
Consequently the reported FP32 errors include input-quantization differences
and cannot isolate encoder arithmetic. The BF16-to-BF16 comparison is not
affected by this confound. `scripts/qwen35_vision_reference_audit.py` prepares
a separate audit using identical BF16 input patches promoted to FP32 only
for reference arithmetic. It freezes the original screenshot and two held-out
screenshots, hashes reference tensors and code, and retains the original gate
outcomes. Results require interpretation; collecting them does not itself
qualify visual compatibility. This diagnostic is queued after the text pilot
to preserve exclusive GPU timing.

After the first 9B pair, an independent same-weight text diagnostic compared
qw3 and llama.cpp on five fixed prompts per model (5, 23, 25, 1,145 and 2,325
tokens), including a 27B control. All 15 first-token argmaxes agreed. Top-32
set overlap was 31–32/32 for 4B, 28–32/32 for 9B and 31–32/32 for 27B;
logits were not identical. Common-token centered logit RMS differences ranged
from 0.046–0.176, 0.020–0.230 and 0.038–0.425 respectively. This supports
continuing the text pilot without evidence of a large new shape/loading
error in these cases; it does not prove long-trajectory accuracy or MTP
equivalence. The diagnostic uses FP16 KV and no speculation; the frozen pilot
continues with FP8 KV/MTP=4. Raw dumps, commands, identities and the review are
under `reference_logits_audit/`. Partial UTF-8 display-token bytes are retained
in the dumps and decoded with surrogate escaping only by the reader;
comparison uses integer token IDs and numeric logits.

The new `qw3-vision-attention-oracle` independently checks 64/72-dimensional
vision attention against FP64 CPU softmax, including two separate images,
an empty segment and 128-lane padding. Both cases passed (maximum absolute
error 0.002032/0.002015; RMS 0.000498/0.000495). The fixture also distinguishes
incorrect padded-dimension scaling and cross-image attention. This narrows
the unresolved full-encoder visual difference; it does not waive that gate.

The first 9B pair is also retained as a quality failure. Ordinary qw3 had
8/80 new tests and 26/116 existing tests pass, versus 0/80 and 21/116 for
KVMem. Ordinary execution took 1,334.45 seconds, generated 160,697 tokens and
had one native compaction (client-reported 57.129 seconds); KVMem took 612.99
seconds, generated 69,583 tokens and did not reach semantic reselection.
This is not evidence of a 2.18x maintenance-induced speedup: generated work
and quality differ, and compaction accounts for only 4.28% of ordinary agent
time. Both processes and verifiers completed normally. The model/engine
artifacts and all completed result hashes were preserved through the
reference-audit pause, then the remaining 9B-first schedule resumed.

The fifth 9B KVMem trajectory (`drizzle-orm-window-function-builders`)
exercised both refresh routes. Request 42 reached the reserve guard while
generating a long, unfinished `Write` call. The existing emergency path
retained the full open fragment and used the original query; selection took
871.661 ms inside decode, with no newly generated query. Request 72 later
generated a 123-token query in 1,202.259 ms, selected 4,096 of 8,051 blocks
in 142.447 ms including KV organization, and replayed 991 tokens from aligned
checkpoint P=256640 in 201.204 ms. Its server visible-content TTFT was
2,012.625 ms. Both requests completed and execution continued; the final
verifier later reported 0/130 feature tests and 566/566 existing tests passed.
`first_real_refreshes_9B.json`
preserves the source lines and timing boundaries.

The pilot report separates generated-query refreshes, original-query
fallbacks, direct queries and replay requests. The legacy `guided_query_count`
also counts fallback log records and must not be read as a count of newly
generated queries. Engine decode time includes mid-decode maintenance;
its logged query/reselection totals are exported separately and aggregated
once per request, because individual mid-decode event timers are cumulative.

Historical 27B data for all eight preselected task IDs and seed 1000 are
aligned under `reference_27b_matched/` (16 arms). The exporter checks the
recorded task entries, sampling settings, policy, MTP=4 and Claude executable
against this pilot, and verifies FP8 KV from the old server logs. Both old
policies passed 2/8 full tasks. These are historical results: engine and
harness revisions differ, and the export does not establish hardware
equivalence. Keep them separate from the current-build 4B/9B measurements;
their timing differences cannot isolate model size or current engine speed.

The fixed 9B text pilot subsequently completed all 16 arms (eight paired
tasks, seed 1000), with normal agent/verifier termination and clean model
service protocols. Full-task success was 0/8 for each policy. Mean agent
time was 655.20 s for ordinary qw3 and 759.56 s for KVMem; weighted engine
decode throughput was 128.14 and 126.62 token/s. Mean prefill time was 36.09
and 22.73 s, while mean decoded tokens were 68.40K and 82.07K. Thus lower
prefill time did not yield lower overall agent time in this pilot; different
generated work and task failures prevent a maintenance-induced speedup or
ability-preservation claim. Ordinary execution had two native compactions;
KVMem had four generated-query refreshes with replay and one original-query
fallback inside decode. The completed subset, source metric hashes, MTP
counters and validated per-request timings are in `9B_completed/` (3,124
request rows). The remaining 4B pilot and visual qualification continue
separately; completing the 9B subset does not complete the overall goal.

A CPU-only analysis of saved BF16 visual tensors localized the 4B divergence:
the 20 largest-error rows account for 80.3% of squared error after block 10
and 91.4% after block 11 (8,160 rows total). The six largest-error block-10
rows have bitwise-identical initial patch/position embeddings between the
two encoders. Their block-9 output variances are 0.105--0.184, in roughly
the top 2% of rows and far above the LayerNorm epsilon of 1e-6. Thus near-zero
input variance does not explain this particular amplification. This does
not exclude an indirect effect of earlier position-rounding differences
through attention. It motivates separating attention and MLP contributions
if the queued position ablation does not explain the discrepancy.
`vision_error_locality.json` records source hashes and row statistics;
`vision_block10_input_variance_probe.json` records the independent FP64
normalization probe. Neither diagnostic changes or waives the visual gate.

`vision_same_block_input_audit.py` prepares a further reference-only diagnostic:
each of the 24 reference blocks receives the exact saved native BF16 input to
that block, independently of the previous reference block's output. Comparing
native/reference local errors against math/flash reference spread can separate
local operator discrepancies from accumulated propagation. Input shapes and
source hashes passed CPU validation, including equality of the saved native
final output with the frozen parity output. GPU execution remains pending;
these deliberately disconnected block outputs must not be presented as a
full encoder result or used for agent inference or performance measurements.

An interim MTP comparison restricts both models and policies to the first three
completed task IDs (12 arms). Weighted draft acceptance is 51.01%/52.60% for
4B ordinary/KVMem and 77.14%/76.09% for 9B. That corresponds to 2.04/2.10
accepted drafts per batch for 4B versus 3.08/3.03 for 9B. Observed engine decode
throughput is 127.89/124.89 and 129.56/128.26 token/s, respectively. Lower 4B
draft acceptance is a relevant observed difference when interpreting why
real agent throughput does not follow parameter count alone. These are still
different generated trajectories, not an isolated model-size or MTP-speedup
experiment. `mtp_acceptance_matched_tasks.json` preserves the 12 source rows
and server-log hashes; phase timers do not identify exclusive CPU/GPU costs.
