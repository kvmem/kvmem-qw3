# Computer-use acceptance evidence index

Status: **active; not a completion certificate**. This index separates the
scope established by each result from the remaining required work. The user
retired the initial 48GB GPU cap; the configured local GPU has 97,887 MiB.
The original [goal and acceptance requirements](computer_use_codex_goal_20260930.md)
remain in force with that explicit scope update.

The companion [machine-readable inventory](computer_use_acceptance_20261001.json)
records hashes of the retained result files inspected for this review. Runtime
paths below are relative to `apps/computer_use/.runtime/`. A running suite's
partial count is a time-stamped observation, not its final result.

| Requirement | Evidence inspected | Scope and current result |
| --- | --- | --- |
| Preserve the pre-existing implementation | Git checkpoint `d968ebc`; later application commits through `53cd0bf` | Existing DFlash/KVMem/tokenizer work committed. Model files and private/bulk artifacts excluded. |
| CPU build and baseline checks | `cpu-baseline-recheck-01/{result.json,ctest.xml,check-0.log,check-1.log}` | Release CPU build from M0 rechecked: 44 CTests and 2 benchmark-helper tests passed. C++/original helper sources unchanged from checkpoint. Prior logs retained before replacing CTest's last-run log. |
| Codex Responses subset, image/tool transport and lifecycle contracts | Native client bundle 04 manifest; `windows-native-suite-02/on-notepad-0/native-client-tests*` | 31 native Windows tests passed with actual pinned Codex. Its 19 bundled source files match the frozen baseline; modified clients need fresh qualification. This is subset compatibility, not the full Responses API. |
| Actual local model/image/MCP loop | `codex-gpu-smoke-02`, `dflash-smoke-04`; native pilot/task artifacts | Actual QW3 vision and DFlash exercised, rather than fixture-only model responses. [Provider report](computer_use_codex_m1_20261001.md). |
| Screenshots, bounded actions, stale observations and no-progress handling | Native contracts; `windows-scroll-contract-03`; `windows-recovery-03/result.json` | Real Windows scroll/failure/recovery evidence plus contract coverage. The old executor recovery evidence does not qualify the newly introduced clipboard critical section. |
| Literal text and correct saved files | `windows-typing-paste-04/result.json`, `windows-clipboard-01/check-final.json` | Final executor: four Notepad inputs plus one Edge Unicode/4096-character form passed; five clipboard fixture formats restored. Existing failed approaches retained. |
| User controls and UI | `browser-ui-check-01`; `windows-native-type-pilot-01/visual-review.json` | Actual browser start/pause/refresh/resume/stop with scripted endpoints passed; native real-task UI also visually inspected. No JavaScript errors in the button test. |
| Pause and process restart | `windows-restart-01/result.json`; `windows-native-restart-01/summary.json` | Linux and Windows clients restored the same task and stayed paused until explicit resume; real files verified. Forced executor termination while it owns the temporary clipboard remains open. |
| KVMem on/off × none/MTP/DFlash fixed short inputs | `comparison-short-02/{summary.json,output_checks.json}` | 54 completed requests passed the limited next-tool/filename checks. Same frozen inputs, prefix reuse disabled. No desktop actions dispatched by this replay. |
| Above-budget historical visual evidence | `comparison-long-02/visual-checks.json`; `comparison-long-controls-01/visual-checks.json` | 12/12 positive/changed-image-negative cases across all six profiles passed at 158,911 tokens. One repetition per case, not a statistical reliability claim. |
| Growing history and Codex retention | `long-history-probe-01`; `codex-long-01/history-check.json` and raw inference/events | Growing fixture history and actual Codex preservation of eight images/4,800 notes demonstrated. This does not substitute for the live desktop archive task. |
| Resource and speculation accounting | [Long matrix](computer_use_long_matrix_20261001.csv), [memory diagnostic](computer_use_memory_diagnostic_20261001.csv), scoped raw reports | DFlash accepted/drafted/fallback counters, TTFT, task/action and process/device measurements retained. Allocation traces passed on both policies but explicitly exclude some allocation classes; not a complete census. |
| Three repetitions of each native Windows core task, both policies | `windows-native-suite-02/summary.json`; frozen `qw3-type-20261001` | Completed 18/18 with a passing independent evidence audit. Same source/model/binary/snapshot identities across both policies; per-trial CSV retained. This qualifies the frozen clipboard-preserving executor, not the undeployed recovery candidate. |
| Actual long desktop task across retrieval threshold, both policies | [Completed repaired pair](computer_use_windows_long_pair_04_20261001.json); `windows-long-{on,off}-04` | On completed 36/36 pages in 644.65s; off timed out at 901.80s, last visible page 18/36. Both used the same 900s deadline and frozen configuration. The comparison is complete; both tasks did not pass. Earlier paired failures remain retained. |
| Same-turn old-image positive/negative controls | [Failure report](computer_use_same_turn_visual_01_20261001.json) | At 162,062 tokens, on returned a wrong reference/name; off was correct. Both changed-image negatives returned null, but on produced no semantic retrieval dump. Guided-refresh configuration requires investigation. |
| macOS executor/client and physical targets | [Mac access preparation](computer_use_mac_test_access_20261001.md) | Still open. Readiness helper prepared; only syntax/help/non-Mac refusal verified. No Mac or physical Windows desktop acceptance has run. |

## Comparison interpretation

Live desktop trials compare deployed configurations: KVMem on also enables
prefix reuse; off does not. Their screenshots and action trajectories can
therefore differ. The fixed-input 54-case short matrix and 12-case long matrix
explicitly disable prefix reuse on both sides. Do not attribute all live-task
latency differences to sparse retrieval alone. The long replay has one
repetition per positive/negative case; it does not establish a reliability rate.

An isolated input candidate now replaces the focused native edit selection
without temporary clipboard ownership. It remains outside the deployed source:
no Windows runtime checks have been performed on it. The completed matrix
qualifies the frozen executor hash below, and the clipboard interruption gate
remains open. A native selection/undo/read-only/NUL probe is prepared alongside
the candidate; the original paired queue is now terminal.

## Frozen Windows qualification identity

- Codex 0.159.2; Windows CPython 3.13.13; client bundle 04 SHA-256
  `4cf9d3cf8e5d8cad9c2d5ec4e4e20c7cb24664dc500a2d1fef4d279b8d798945`.
- Qwen3.8-27B Q8_0, FP8 KV, matching BF16 DFlash 2; engine SHA-256
  `e14b429817f42cedee5659c786087f0265f6335a58073d884a2108782ca9a2ad`.
- Executor SHA-256
  `b5e4eafa2204609a32914ef90bd2aaec85a902884ec7199ea2e3fbb8bddd37fb`.
  The snapshot retains unchanged fixture/bootstrap hashes and a cleared test
  clipboard. The queue checks all frozen application-source hashes before each
  stage; code changes must not be slipped into the running comparison.
- App, Codex and MCP run natively inside Windows; model inference runs on the
  physical Linux GPU host through the private authenticated relay. This proves
  an entirely local single-host VM deployment, not native Windows GPU serving
  or Mac-local inference.

## Open gates

The repeated native matrix passed for its frozen executor. The goal cannot
close until the same-turn visual retrieval failure is resolved, the clipboard
ownership failure case is qualified or repaired, and the macOS/accessible
physical-desktop requirements are satisfied. The live on/off archive comparison
is complete, including the off-policy timeout.
Earlier failures remain failures even when their repaired pilot passes.
Missing hardware is recorded as missing evidence, never a successful test.
Reproduction commands live in the [application README](../apps/computer_use/README.md).


## Long-history repair validation

The original four-subblock generated-tail failure was reproduced against the
original library. The revised tail finalizer retains raw-row and shape checks,
and the CUDA capture kernels clear unused tail subblocks in recycled storage.
The rebuilt CPU suite passes all 44 CTests. Both CUDA index/merge tests pass,
covering CPU/GPU indices, FP32/FP8 raw keys,
partial extension, missing-row refusal and typed mean output. The application
now passes 34 Linux contracts with actual Codex enabled. Native bundle 04
remains evidence for its frozen source, not this modified client.

The [retained-request continuation audit](computer_use_continuation_repair_20261001.json)
replays one 130,005-token request followed by 133,278, 136,554 and 139,827 tokens.
All four completed. The three above-budget requests reused 130,174, 133,454 and
136,724 tokens, respectively, and completed in 8.18, 8.26 and 8.50 seconds.
Their own tagged retrieval records report ready query/index state, no fallback
and 1,019/1,042, 1,022/1,068 and 1,023/1,093 selected/total blocks. Historical
message groups match the expected 77/79/81/83, including older same-turn tools.
This diagnoses continuation and mapping; no desktop actions were dispatched,
and it does not establish final old-image answer quality or live task success.


The first repaired live attempt (`windows-long-on-03`) was interrupted by
SIGTERM after 66 completed actions, before its 900-second deadline. The signal
sender is unknown; no kernel OOM/GPU fault was found in the inspected interval.
Queue, app and engine termination were verified, then the dedicated desktop
was explicitly paused. There is no final task result and no off-policy run;
this is incomplete infrastructure evidence, not a passed task or a deadline
failure. [Interruption record](computer_use_archive_interruption_20261001.json).

The repaired pair (`windows-long-{on,off}-04`) finished under transient user
unit `qw3-computer-long-repair-02.service`. Both task deadlines were 900 seconds,
with the same snapshot and `c436ec6` app/engine sources. Its exit cleanup passed,
and the VM ended paused. Independent comparison checks passed; the on task
passed and the off task timed out. The unit is terminal.


## Repaired live archive, KVMem on

`windows-long-on-04` completed the actual 36-page Windows workflow in 644.65s
under the unchanged 900-second deadline. All 81 actions completed, with no
rejections or unresolved receipts. The independently checked 363-byte
`archive-review.json` contains the correct reference and pages 1 through 36;
SHA-256 is `da215048cb633ff2b47051629299f45ffc52084165b3368cf65e03c0673b10be`.
The retained final UI also shows the correct reference and successful save.

The request that first typed the reference contained 256,966 prompt tokens and
80 images, preserving all 79 distinct prior images. The reference was absent
from prior text. Its exact request tag matches an as-run retrieval record with
1,021 of 2,009 blocks selected, query/index ready and no scorer fallback.
Across the task DFlash accepted 11,071 of 19,164 drafted tokens, with zero
fallback requests. Sampled whole-device GPU peak was 45.785 GiB; engine RSS
high-water was 40.373 GiB. These resource scopes must not be added together.

[Independent audit and identity](computer_use_windows_long_on_04_20261001.json).
The client/model ran on Linux and actual desktop actions ran in Windows on
the same host. This is one successful task, not a reliability estimate or
native Windows GPU qualification. The completed pair is summarized below.


## Completed paired comparison and remaining visual failure

| Measurement | KVMem on | KVMem off |
| --- | ---: | ---: |
| Task outcome, 900s limit | Passed, 644.65s | Timed out, 901.80s including stop overhead |
| Last verified progress | 36/36, exact saved JSON | Last visible page 18/36, no verified output |
| Completed actions | 81 | 40 |
| Completed inference requests | 82 | 40, plus one cancelled request |
| Bridge inference elapsed median / p95 | 8.16s / 14.10s | 20.53s / 43.01s |
| Max completed prompt tokens | 262,656 | 128,283 |
| Sampled whole-device GPU peak | 45.785 GiB | 67.668 GiB |
| Engine RSS high-water | 40.373 GiB | 32.027 GiB |

[Pair audit and metrics](computer_use_windows_long_pair_04_20261001.json) and
[all request timings, including cancellation](computer_use_windows_long_requests_20261001.csv).
Trajectories differ and on includes prefix reuse. These are one paired task's
observations, not an isolated retrieval speedup or minimum-memory estimate.

The subsequent same-turn visual probe exposed a separate quality failure:
with identical 162,062-token inputs, on returned `QW3-WIN-2026-001` and
`Alice Johnson` instead of the screenshot's `QW3-WINDOWS-2026-ALPHA` and `Alice`;
off returned the correct values. Both changed-image negative outputs were
null. On suppressed semantic reselection for the existing user query and used
pressure-prefill; no retrieval dump was produced. The diagnostic also reports
its missing-file validation error. This failure remains open, and does not
become a pass because the earlier live archive succeeded.
[Full retained failure](computer_use_same_turn_visual_01_20261001.json).
