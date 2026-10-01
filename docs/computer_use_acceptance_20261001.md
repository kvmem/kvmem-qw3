# Computer-use acceptance evidence index

Status: **active; not a completion certificate**. Windows VM functionality and
both fixed-input decoder matrices are qualified at the identities below. The
long live comparison is complete: KVMem on passed; off timed out. Checkpoint
latency, native-client resource attribution, vision timing, macOS and physical
hardware acceptance remain open.

The user retired the initial 48GB GPU limit. Tests use the current host's RTX
PRO 6000 Blackwell Server Edition, reporting 97,887 MiB. Inference is local on
that Linux host. Native Windows client tests run the application, Codex and MCP
in its Windows VM. This does not qualify native Windows GPU inference or an
ordinary Windows/Mac computer's capacity.

The [goal](computer_use_codex_goal_20260930.md) defines the full acceptance scope.
The [JSON inventory](computer_use_acceptance_20261001.json) and linked reports
retain identities and distinguish passed functionality from open measurements.
Runtime paths below are relative to `apps/computer_use/.runtime/`.

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

Independent analysis and documentation commits occurred during the frozen
experiments. Per-trial Git HEAD can differ; application, binary, input, bundle
and snapshot hashes identify what was tested. No current GPU diagnostic may
have its frozen sources or engine replaced while it is active.

## Completed current checks

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
`qw3-computer-qualification-followup-01.service` is separately running the
unchanged-engine prefix reproduction, followed by standalone vision profiling.
Read its live unit and result before starting GPU work or replacing sources.

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

## Remaining work and retained failures

1. **Checkpoint latency:** four on-policy archive requests lost an admissible
   checkpoint after a late periodic selection, causing 61–108s cold prefills and
   stale observations. [Cause and isolated candidate](computer_use_guided_prefix_diagnostic_20261001.json).
   The unchanged-engine 50-request replay is running. The scheduling candidate
   passed CPU policy checks but is not applied or GPU-qualified. Keep the replay
   alignment guard and 60s observation lifetime.
2. **Native-client resource attribution:** [defect and prepared probe](computer_use_native_resource_attribution_20261001.json).
   Validate PID plus creation identity on actual Windows, then run a fresh matched
   resource pair. Eight offline tests and probe readiness are not native API or
   application resource acceptance.
3. **Vision timing:** [standalone probe readiness](computer_use_vision_profile_readiness_20261001.json).
   Cold encoding, one new image, cached reorder and CPU/GPU embedding parity
   are queued. Server validation time includes other work and is not a pure
   image encoding timer. Nested phase timings must not be double counted.
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
