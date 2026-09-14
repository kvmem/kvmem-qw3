# Motivation v3: reproducible scheduling measurements

## Objective and evidence boundaries

Audit the 49 historical SWE/OpenHands trajectories, collect new model-native
attention across diverse inputs, and produce multi-trajectory scheduling figures.
The historical live cohort and the new replay cohort are reported separately.

The 49 historical trajectories correspond to 343 uniquely matched requests;
all were stopped by a six-iteration agent cap and were not officially scored.
Legacy token prefixes are missing, so a positional intersection cannot prove
identical historical content. Legacy plots are diagnostic controls.

The frozen new manifest contains 80 distinct source episodes and 957 consecutive
recorded model-call prefixes: 20 software-agent tasks, 20 structured tool/state
episodes, 20 conversations, and 20 GUI tasks. New model continuations are sampled
for each recorded input; their actions are not executed, and the next input uses
the original recorded history. This is counterfactual input replay, not a fresh
closed-loop success-rate benchmark. Never count sampled windows as independent
trajectories or claim 80 new solved tasks.

## Frozen collection protocol

- Selection seed: 20260911; no attention-based selection.
- Source checksums and exact selected frame indices: `new80/manifest.json`.
- Model for all four new domains: Qwen3.8-27B-Q8_0, FP16 KV, context cap 65536.
  Using one vision-capable model avoids confounding new domain with model.
  The legacy Qwen3.6 cohort is not pooled with the new cohort.
- Full-visible dense attention; KVMem and MTP disabled.
- Prefill queries: final 64 prompt tokens, read from the unmodified batched
  prefill Q/K using a causal attention-mass side probe. They are a
  prefill-tail diagnostic, not the compact-index production retrieval score.
- Generated-token cap: 2048; thinking budget: 512; temperature: 0.
  Short generations stay short and length truncations are reported.
- Trace: every token, 128-token historical blocks, every standard-attention layer.
  Raw head-summed per-layer rows are preserved; analysis averages layers.
- Archive actual execution token IDs and mRoPE positions after visual input
  expansion; verify image hashes before using visual prefix identity.
- Primary matched windows: 16/32/64, non-overlapping; 128 is additional.
  Report eligible trajectory/boundary counts separately for each window.
- For a strict boundary pair, both steps need two full windows. Compare the
  across-boundary windows and adjacent within-step windows over the SAME common
  historical support, using the SAME two steps.
- Only complete common historical blocks are candidates. Exclude the first
  sink block and eight recent prompt blocks; report historical total mass.
- Coverage: select once from prefill-tail attention; evaluate only later decode
  rows. Compare against a per-token oracle at exactly the same block budget.
  Main cross-domain budget is 10% of candidate blocks, rounded up; also report
  5%, 20%, and absolute 8/16/32/64-block budgets.
- Aggregate windows to steps and steps to trajectories before across-trajectory
  averaging. Bootstrap trajectories (5000 replicates, fixed seed). Shaded 95%
  bootstrap confidence intervals are distinct from token or task quantiles.
- Do not infer task quality, production retrieval latency, or million-token
  scalability from attention coverage alone.

## Source selection

Software: existing official Pier/mini-swe-agent DeepSWE recordings. Select one
eligible source per task by sorted file order, not reward. Keep consecutive
assistant-prediction prefixes after at least 4000 serialized input characters,
bounded at 190000 characters for collection feasibility.

Tool/state: local AgentLongBench 32K records, sampling distinct base episode IDs
across masked/unmasked and concise/verbose forms. Variants of one episode share
an independence unit. Most verbose records do not contain eight eligible
consecutive steps; quotas are redistributed before measurement, and the frozen
manifest records the actual variants.

Dialogue: recorded LongMemEval-S haystack conversations, not the final QA score.
Full source-conversation hashes prevent exact trajectory duplication.

GUI: OSWorld recorded successful API exchanges with restorable processed input
images. Choose the longest uninterrupted eligible recording per distinct task,
then sample task IDs with the fixed seed. This favors recordings with enough
steps and available artifacts; the resulting selection limitation must be
reported. A successful API exchange is not a successful GUI task.

Source projects: [AgentLongBench](https://github.com/euReKa025/AgentLongBench),
[LongMemEval](https://github.com/xiaowu0162/LongMemEval), and the local pinned
OSWorld/Pier/mini-swe-agent configurations archived in this repository.

## Reproducibility and execution

Run scripts with the repository `.venv/bin/python` (NumPy/Matplotlib installed).

```bash
.venv/bin/python scripts/motivation_v3/audit_legacy.py --out RESULTS/legacy49
.venv/bin/python scripts/motivation_v3/analyze_legacy.py --audit RESULTS/legacy49/audit.json --out RESULTS/legacy49/analysis
.venv/bin/python scripts/motivation_v3/build_probe.py --out RESULTS/probe
.venv/bin/python scripts/motivation_v3/prepare_dataset.py --out RESULTS/new80
.venv/bin/python scripts/motivation_v3/serve_probe.py --out RESULTS/server --binary RESULTS/probe/build/qw3
.venv/bin/python scripts/motivation_v3/collect.py --manifest RESULTS/new80/manifest.json --out RESULTS/measurements --trace RESULTS/server/attention.jsonl --input-trace RESULTS/server/inputs.jsonl
.venv/bin/python scripts/motivation_v3/analyze_new.py --manifest RESULTS/new80/manifest.json --measurements RESULTS/measurements --out RESULTS/figures
```

Use absolute RESULTS paths or run from repository root. The server is a separate
long-running process; start the collector in a separate terminal. The builder
requires CUDA 13 and installed FlashInfer headers; adapt the architecture flag
to another GPU. The probe source is a snapshot and does not modify production
sources or overwrite the existing build. The vision worker is also snapshotted.

Collection waits for model readiness, dispatches serially, associates exact
trace byte intervals to requests, preserves failures, and stops on five request
failures, ambiguous timeout, disk pressure, or the raw-trace size cap. It never
deletes data to free space. Completed steps are resumably skipped; failed steps
require a deliberate diagnosis/retry into retained artifacts.

Initial non-FlashInfer smoke attempts were infrastructure failures. A scalar
prefill-tail pilot completed eight requests but changed greedy continuations
relative to ordinary batched prefill; it is excluded from the formal cohort.
The revised probe reads already-computed batched Q/K without changing batch
shapes or KV/recurrent-state advancement. All pilot artifacts are retained.

The native chat template may remove earlier reasoning as new queries arrive.
Exact prefix checks therefore bound the usable common history; matching source
message arrays alone is insufficient. Candidate-history size and total mass
must accompany the coverage percentages.

`schedule_diagnostics.py` adds offline periodic-update controls (previous 32
queries, refresh intervals 32/128/512), context/decode-length strata, and natural
versus truncated endings. Update counts are algorithmic counts, not runtime.
These controls use full-visible attention, not production compact scores.

## Resumed campaign (2026-09-11)

The active run is `benchmark/results/motivation_v3_20260910/formal80`, launched
with `run_campaign.py` and the `probe_batched/build/qw3` binary. The revised
probe passed eight request alignment/causality audits and exact raw-output
comparisons against ordinary batched prefill for a text and an image request;
see `probe_verification.json`. All pilot requests are excluded from formal
counts. The campaign runs four balanced batches of 20 source trajectories,
restarts its own server between batches, records GPU memory every two seconds,
and generates the main and control figures after each batch. The selected
inputs are frozen in `new80/manifest.json`; a launch-time protocol snapshot
is retained under the campaign directory. A finished collection pass can
still contain individually retained failures, so final sample counts always
come from request status artifacts, not the campaign label.


Final independent checks and the paper-facing Chinese report:

```bash
.venv/bin/python scripts/motivation_v3/audit_new.py --manifest RESULTS/new80/manifest.json --measurements RESULTS/formal80/measurements --out RESULTS/formal80/integrity_audit.json --require-complete
.venv/bin/python scripts/motivation_v3/write_report.py --campaign RESULTS/formal80 --manifest RESULTS/new80/manifest.json --legacy RESULTS/legacy49/analysis/analysis.json
```

The report explicitly distinguishes partial snapshots from a completed cohort.
One-trajectory strata retain their point estimate but have no population CI.
The main figure uses a 7.2-inch double-column layout with shared JS axes.
The running campaign invokes analysis after each batch; independent audit and
paper-report generation above are additional final completion gates.


Decode-query accounting follows the native no-client-stop, non-speculative
loop: the first visible output comes from prefill logits. A natural EOS-ending
request has one decode query per visible output token, but an output-cap-ending
request has one fewer query, because its last output is not forwarded again.
The independent audit checks this exact distinction. Cap-hit statistics use
actual completion count as well as the finish label: the serving API can replace
`length` with `tool_calls` when a tool frame was parsed. Natural and cap-hit
requests are reported separately in the length/control tables.


Layer robustness is generated automatically by `analyze_new.py` via
`analyze_layers.py`. It reads the first/last matched query windows directly
from raw per-layer rows for complete trajectories, reports all 16 standard
attention layers, and caches results keyed by inputs and analysis source.
`layer_reconstruction_check.json` independently checks that averaging the raw
layers reproduces the primary metrics within 1e-6. Heads were already summed
by the native tracer, so this control cannot establish per-head stability.

`natural_end_only` in the primary summary repeats matched-boundary statistics
with cap-hit requests removed. Source-step adjacency remains mandatory;
removing a request never joins its neighbors into a new boundary. The main
results retain all collected requests; this is a sensitivity control.

`refresh_distance_controls.py` is also invoked automatically by `analyze_new.py`.
It adds absolute-query-distance coverage, with full-bin eligibility counts,
and a stale-set control. The latter retains the actual previous prefill's
selected block IDs only where they survive the current verified prefix; it
does not refill invalid slots. Current-prefill refresh and oracle match that
surviving count, so this paired effective budget is distinct from the main
current-step 10% budget. The last absolute bin ends at 2047 forward queries,
matching native capped-output accounting. These controls are offline and
do not alter collection, source sampling, or the production selector.

`write_paper_draft.py` runs after `write_report.py` during finalization. It
builds `paper_design_draft.md` from the saved main and paired scheduling
statistics. It requires error-free audit evidence covering the exact analyzed
trajectory IDs and request counts, verifies those IDs against the primary
manifest, and rejects partial trajectories. It reports interim status whenever
the full primary cohort is not ready. The draft includes measured confidence
intervals, window exceptions, scope limitations, and the metadata-driven cohort
amendment; `paper_design_draft_evidence.json` records input and script hashes.
Use `--out` to generate a review copy without replacing a manually edited draft.
Generating this draft does not replace final figure and interpretation review.
