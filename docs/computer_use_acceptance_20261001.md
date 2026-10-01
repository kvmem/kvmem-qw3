# Computer-use acceptance evidence index

Status: **active; Windows VM evidence reviewed, Mac and physical targets open**.
Baseline engine `005654a6...` has qualified native Windows core, restart and
resource checks. The repaired engine `9f1d1515...` passed its live KVMem-on
archive task and both fixed-input decoder matrices (54 short and 12 long
checks). Its paired off-policy archive timed out under the same deadline.
The supervised queue and cleanup are complete. These results do not qualify
native Windows GPU inference or Mac/physical desktop deployment.

The user retired the initial 48GB GPU limit. Tests use the current host's RTX
PRO 6000 Blackwell Server Edition, reporting 97,887 MiB. Inference is local on
that Linux host. Native Windows client tests run the application, Codex and MCP
in its Windows VM. This does not qualify native Windows GPU inference or an
ordinary Windows/Mac computer's capacity.

The [goal](computer_use_codex_goal_20260930.md) defines the full acceptance scope.
The [JSON inventory](computer_use_acceptance_20261001.json) and linked reports
retain identities and distinguish passed functionality from open measurements.
Runtime paths below are relative to `apps/computer_use/.runtime/`.

The next Windows expansion is a [10-scenario OSWorld-derived evaluation plan](computer_use_osworld_windows_10_20261001.md),
with a [pinned source/asset and proposed run manifest](computer_use_osworld_windows_10_20261001.json).
It is **planned, not new task-pass evidence**. The [measurement protocol](computer_use_windows_complex_eval_20261001.md)
defines outcome, latency, resource and recording comparisons. A synchronized
offline replay of retained archive observations is available in
`comparison-replay-long-07-04/index.html`; it is screenshot reconstruction,
not continuous video or a result from the proposed ten scenarios.
The [execution and agent-optimization plan](computer_use_windows_execution_optimization_20261001.md)
adds platform readiness gates, failure analysis, separate V0/V1 workflow versions,
and a four-condition retest matrix. Its status ledger contains planned cells only.

## Current qualified identities

- Application content: `7d18e50`; engine SHA-256
  `005654a65d2341ca8b6a504319a6b2b456e9901b9dd5adf6693a4335cc6633ea`.
- Target Qwen3.8-27B Q8_0, FP8 KV; matching BF16 DFlash2. Native MTP and
  target-only decoding are separate controls. [Model asset identities](computer_use_model_assets_20261001.json).
- Windows client bundle 06: `23851fd5c83e8b4e70a2595fcf46d3f1c7f1f600e1ff140a3ade8b2ad69c08f4`;
  Python 3.13.13, Codex 0.159.2, snapshot `qw3-exact-20261001`.
- Executor: `6678710b6425f3d1a1828e0ba90213590526c9adad2b2751fb9d5f317747b45f`.
  [Bundle and snapshot preparation](computer_use_windows_baseline_06_20261001.json).
- Live on-policy tests explicitly use guided refresh `both`, retrievable visual
  history and prefix reuse. Fixed-input controls disable prefix reuse and guided
  refresh. These profiles answer different questions and must remain separate.

The workspace engine `9f1d1515...`, from repair commit `1fb9dbb`, passed 44 CPU
and three targeted policy/CUDA checks, the matched 50-request prefix replay,
eight seed/long visual checks, and the live/fixed comparisons below. Earlier
native Windows measurements belong to engine `005654a6...`, retained in
`guided-boundary-candidate/qw3-baseline-005`.

`qw3-computer-guided-live-01.service` is terminal with MainPID 0. It ended
failed because the dense archive exceeded the deadline; all comparison stages
were attempted, both fixed matrices passed, and cleanup left the desktop
paused. This is a completed comparison with a retained task failure.

Independent analysis and documentation commits occurred during the frozen
experiments. Per-trial Git HEAD can differ; application, binary, input, bundle
and snapshot hashes identify what was tested. No current GPU diagnostic may
have its frozen sources or engine replaced while it is active.

## Retained baseline checks

| Requirement | Evidence | Result and scope |
| --- | --- | --- |
| Original work preserved and host build | Checkpoint `d968ebc`; `cpu-baseline-recheck-01`; repair build logs | 44 CPU CTests, 2 benchmark-helper checks; repaired CUDA index/merge tests passed. |
| Responses subset, images, tools, cancellation and lifecycle | Bundle 06 and native trial logs | 34 contracts with actual Codex, no skips. This is the implemented subset, not full Responses compatibility. |
| Native Windows core tasks | [Matrix 04](computer_use_windows_native_04_20261001.json), [18 trials](computer_use_windows_native_04_trials_20261001.csv) | 18/18: Notepad, browser form and cross-app copy, three repetitions per KVMem policy, independently verified output files. |
| Native application crash/resume | [Restart 02](computer_use_windows_native_restart_02_20261001.json) | 2/2 on/off; persisted task/thread, paused restart, explicit resume, correct copied file. |
| Input ownership and uncertain delivery | [Native edit and recovery](computer_use_native_edit_recovery_20261001.json) | Native edit control semantics and exact input probes passed; forced dispatch death stayed paused and required explicit reconciliation without duplicate insertion. |
| Short fixed-input controls | [Controls 03](computer_use_short_controls_03_20261001.json), [54 requests](computer_use_short_controls_03_requests_20261001.csv) | 54/54 limited next-tool/filename checks: none/MTP/DFlash × on/off × three inputs × three repetitions. No desktop actions. |
| Long fixed-input controls | [Controls 03](computer_use_long_controls_03_20261001.json), [12 requests](computer_use_long_controls_03_requests_20261001.csv) | 12/12 historical-image positive/changed-image-negative checks at 158,911 tokens. On performed actual sparse retrieval; all requested speculative methods ran without fallback. |
| Same-turn visual continuation | [Guided diagnostic](computer_use_guided_visual_20261001.json) | Four long on/off positive/negative checks and four short seeds passed with guided refresh `both`. Earlier guided-off failure remains retained. |
| Long live archive | [Pair 06](computer_use_windows_long_pair_06_20261001.json), [128 request timings](computer_use_windows_long_requests_06_20261001.csv) | On completed 36 pages and exact saved JSON; off hit the unchanged 900s deadline at page 18. Comparison complete, not two successful tasks. |
| Request phase accounting | [Analyzer scope](computer_use_phase_accounting_20261001.json); `windows-native-suite-04-phases`, `archive-phase-{on,off}-06` | Core 206/206 and archive-on 87/87 complete. Archive-off has 40 complete plus one cancelled request with missing terminal timing. Dense correlation uses serial order and exact token counts, not native trace tags. |

The terminal `qw3-computer-final-qualification-04.service` attempted every
stage and completed cleanup with the desktop paused. It correctly ended failed
because the dense archive timed out; the 54 short and 12 long controls passed.
`qw3-computer-qualification-followup-01.service` is also terminal with success:
50 unchanged-engine requests reproduced the checkpoint miss, then three standalone
vision trials passed. Read live unit states before starting another test queue.

## KVMem on/off observations

The native core task elapsed medians, in seconds, were:

| Task, three trials per policy | On | Off |
| --- | ---: | ---: |
| Exact Notepad document | 36.56 | 67.66 |
| Browser form and saved JSON | 42.77 | 77.41 |
| Cross-app copy and saved reference | 58.88 | 86.87 |

These live tasks follow different model trajectories. On includes prefix reuse;
these numbers are not a retrieval-only speedup. Durations exclude VM reset,
contracts and model startup. Sampled whole-device GPU peaks were 44.016/67.398
GiB on/off. The legacy native-client RAM/CPU samples are **unqualified** because
PID reuse contaminated one process tree; they are withheld from the comparison.
Independent task outcomes, host GPU and engine RSS remain valid.

The [corrected observer's fresh Notepad pair](computer_use_native_resources_20261001.json)
passed on/off with all 32/57 samples qualified. Peak sums of client working
sets were 210.72/214.74 MiB; private bytes were 122.63/126.17 MiB. The samples
contained 3–5 owned Python/Codex/console processes and matched the actual app
PID and creation identities. These counters exclude the Linux model, controlled
desktop applications and VM overhead. One pair is not a memory advantage
estimate, and it does not rehabilitate the old client counters.

The long live pair used the same snapshot and 900s deadline:

| Measurement | On | Off |
| --- | ---: | ---: |
| Outcome | Passed in 820.16s | Timed out; 901.88s including stop overhead |
| Last verified progress | 36/36; exact 363-byte JSON | Page 18/36; no verified file |
| Completed actions | 78 | 40 |
| Rejected actions | 4 stale observations | 0 |
| Max completed prompt tokens | 267,535 | 128,386 |
| Sampled whole-device GPU peak, GiB | 46.127 | 67.674 |

The final reference-producing on request retained all 81 distinct prior images
and selected 1,024 of 2,047 blocks with a ready query/index and no scorer
fallback. The correct saved file's SHA-256 is
`da215048cb633ff2b47051629299f45ffc52084165b3368cf65e03c0673b10be`.
This is one paired workflow, not a reliability estimate.

The fixed long controls remove cross-request prefix reuse. Positive/negative
elapsed times and sampled profile GPU peaks were:

| Decoder | On seconds, positive / negative | Off seconds, positive / negative | GPU GiB, on / off |
| --- | ---: | ---: | ---: |
| None | 64.789 / 62.529 | 60.160 / 59.330 | 38.271 / 61.893 |
| Native MTP | 64.564 / 62.858 | 59.707 / 59.121 | 39.627 / 64.596 |
| DFlash2 | 64.609 / 62.686 | 59.773 / 59.134 | 43.410 / 66.062 |

Each cell is one request, with short answers. On reduced sampled GPU usage but
increased cold inference time in these inputs. The second image case reuses the
profile's image cache, and output token counts vary. No statistical speedup,
allocator maximum or minimum device capacity is inferred from these numbers.
The independent offline audit can be reproduced with
`scripts/computer_use_long_control_audit.py`; the report records all input,
engine, output and retrieval identities.

## Boundary repair: current live and fixed controls

The [new on-policy archive](computer_use_windows_long_on_07_20261001.json)
passed in **416.09s**, with all 36 pages, 80 completed actions, no rejections or
unresolved receipts, and the exact same 363-byte output SHA-256 as the prior
successful run. The final screen shows the correct opening reference and saved
file. Its 81 model requests all have complete, tag-matched phase accounting.

All 41 requests above 131,072 tokens reused history. The longest above-budget
prefill was 1.671s, and no checkpoint miss occurred. Server visible-output latency
was 1.918s median / 4.477s p95; screenshot-to-action latency was 3.942s median /
7.050s p95. Sampled GPU peak was 46.125 GiB. The reference-producing request had
252,987 tokens, preserved all 78 distinct prior images, and selected 1,018 of
1,977 blocks with ready query/index state and no scorer fallback.

The prior guided on-policy run took 820.16s and had four stale-observation
rejections. These are single live runs with different model trajectories;
use the separate matched 50-request replay for controlled latency evidence.
The [matching off-policy task](computer_use_windows_long_pair_07_20261001.json)
hit the same 900s deadline (901.09s including stop overhead), with 39 completed
actions and no rejections or unresolved receipts. Its final paused screenshot
shows page 18/36, unchecked; no result file was produced. It also repeated the
opening reference in an intermediate assistant message, violating the task's
instruction. No successful historical-image recall is claimed for that run.

The paired identities and retained evidence are consistent. On/off sampled
whole-device GPU peaks were 46.125/67.674 GiB. Off has 40 complete model requests
and one deadline-cancelled request without terminal timing; its phase audit
passes consistency, with incomplete coverage. The [six-profile short controls](computer_use_short_controls_04_20261001.json)
passed all 54 limited output checks, with actual speculation and no fallback.
Their sampled GPU peaks were 37.209/62.299 GiB (none), 38.693/65.365 GiB (MTP)
and 42.451/67.541 GiB (DFlash), on/off respectively.
The [long fixed controls](computer_use_long_controls_04_20261001.json) also
passed 12/12 positive/changed-image-negative checks, with identical 158,911-token
inputs, prefix reuse and guided refresh disabled, actual requested speculation
and no fallback. All six on-policy requests had ready query/index state and
actual sparse selection. All 12 requests passed additive server phase accounting;
detailed native phases are available for the eight MTP/DFlash requests. The
four target-only requests lack detailed native-accounting logs, so those fields
remain blank. The four on-policy speculative requests also match native tags;
other phase attribution uses serial order and exact token counts.
[All 122 paired requests](computer_use_windows_long_requests_07_20261001.csv);
[all 81 on-policy phases](computer_use_windows_long_on_07_phases_20261001.csv).


| Decoder | On seconds, positive / negative | Off seconds, positive / negative | GPU GiB, on / off |
| --- | ---: | ---: | ---: |
| None | 64.741 / 62.624 | 60.122 / 59.269 | 38.271 / 61.893 |
| Native MTP | 64.533 / 62.766 | 59.730 / 59.140 | 39.627 / 64.385 |
| DFlash2 | 64.611 / 62.803 | 59.801 / 59.115 | 43.410 / 66.943 |

This is one request per image case/profile, with short answers and an image
cache retained between the two cases. Cold on-policy requests use less sampled
GPU memory but take longer here. These observations are separate from live
history-reuse benefits and do not establish long-output speculation throughput.
[12 request measurements](computer_use_long_controls_04_requests_20261001.csv);
[12 native/server phase records](computer_use_long_controls_04_phases_20261001.csv).

## Qualification boundaries and remaining targets

1. **Checkpoint latency repaired:** four on-policy archive requests lost an admissible
   checkpoint after a late periodic selection, causing 61–108s cold prefills and
   stale observations. [Cause and isolated candidate](computer_use_guided_prefix_diagnostic_20261001.json).
   The [unchanged-engine replay](computer_use_prefix_baseline_20261001.json)
   completed all 50 requests and reproduced the final cold fallback: 61.259s
   prefill and 65.225s client first output. The scheduling candidate is applied and
   built, with 44 CPU and three targeted policy/CUDA checks passed. The
   [matched50-request comparison](computer_use_prefix_boundary_comparison_20261001.json)
   passes: the final request reuses159,104 tokens, reduces prefill61.259s to1.570s
   and total elapsed67.393s to6.897s. The
   [eight visual checks](computer_use_guided_visual_02_20261001.json) also pass.
   The new on-policy real desktop task also passed, with zero checkpoint misses
   or rejected actions. Matched off-policy comparison is complete with a normal
   deadline failure. Both same-engine fixed matrices passed (54 short, 12 long).
   Terminal source/binary hashes matched the frozen manifests and cleanup left
   the desktop paused. Keep the replay alignment guard and 60s observation lifetime.
2. **Native-client resource attribution repaired:** [defect and repair](computer_use_native_resource_attribution_20261001.json).
   The [native probe](computer_use_native_resource_probe_20261001.json) passed
   all ten Windows checks; eight offline validator tests passed. The exact tested
   observer is integrated. The fresh on/off pair passed both task and resource
   audits, with 89/89 samples qualified and the desktop paused at cleanup.
   The old 18-trial client resource comparison remains invalid.
3. **Vision timing completed:** [three standalone trials](computer_use_vision_profile_20261001.json)
   passed CPU/GPU embedding parity. Median first-image GPU processing was
   106.861ms, append-one-new-image 73.681ms, cached reorder 1.242ms. Timings
   include preprocessing after frontend/model construction, with no target or
   draft model loaded. They are not full task latency or cold startup time.
   Nested server/native phases must not be double counted.
4. **Mac and physical targets:** [access/preflight instructions](computer_use_mac_test_access_20261001.md).
   No accessible Mac or physical Windows desktop has been tested. Mac executor,
   client, ordinary-device inference, scaling, IME, focus, permissions and
   takeover remain required. Missing hardware is not a passed test.

Historical reports remain evidence for their original configurations:
[early Windows failures](computer_use_windows_validation_20261001.md),
[original archive timeouts](computer_use_windows_long_pair_02_20261001.json),
[interrupted repair run](computer_use_archive_interruption_20261001.json),
[earlier repaired pair](computer_use_windows_long_pair_04_20261001.json),
[guided-off wrong visual answer](computer_use_same_turn_visual_01_20261001.json),
[missing final newline](computer_use_native_newline_failure_20261001.json) and
[Codex PATH startup failure](computer_use_windows_long_startup_05_20261001.json).
Later successes do not erase those failures or qualify their old profiles.
