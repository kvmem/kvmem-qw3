# Windows repetition and recovery validation

Status: in progress, 2026-10-01. Full application acceptance has not passed.

## First snapshot suite: retained failure

`windows-suite-01` restored `qw3-clean-20261001` before every trial, started a
fresh DFlash engine and host application, and checked files directly through
the guest. Runtime source hashes were fixed across these attempts. The intended
matrix was KVMem on/off × three tasks × three repetitions, with stop on failure.
It stopped at attempt 8: **7 passed, 1 paused before completion**. All attempted
trials used KVMem on; the off trials had not started.

| Task | Repetition | Passed | Task time (s) |
|---|---:|---|---:|
| Notepad | 0 | Yes | 56.11 |
| Browser form | 0 | Yes | 38.49 |
| Cross-app copy | 0 | Yes | 43.95 |
| Notepad | 1 | Yes | 44.48 |
| Browser form | 1 | Yes | 38.91 |
| Cross-app copy | 1 | Yes | 160.48 |
| Notepad | 2 | Yes | 53.31 |
| Browser form | 2 | No | 25.85 |

The slow copy trial repeatedly missed controls in the Save As dialog before
recovering and delivering the correct file. The failed browser trial clicked
`(200,628)` three times while the visible checkbox center was `(198,604)`.
The unchanged-screen guard paused the task. The model never saved the requested
JSON in this attempt. This is a real task failure, not a successful safety-only
acceptance or a reason to discard the sample.

Raw screenshots, actions, full model requests, events and resource samples
remain under `apps/computer_use/.runtime/windows-suite-01/`. Per-trial metrics
are also in `computer_use_windows_suite_01_20261001.csv`. Task time excludes
VM and engine startup. Screenshot-to-action and action-receipt timings use
host timestamps, avoiding the guest's different wall-clock timezone. Quantiles
use linear interpolation within each trial; server visible-output time is
measured by the inference server. This incomplete suite does not establish an
on/off live-task speed comparison.

## Changes under requalification

The Windows observer now provides bounded UI Automation control names,
physical-pixel bounds, centers and focus state alongside the full screenshot.
Only enabled, onscreen interactive controls in the foreground window are
included; password controls are omitted. The model instructions use matching
visible controls' centers. Actions still use the normal mouse and keyboard;
the application does not invoke controls directly or read expected task files
to guide the model. The fixture and task definitions remain unchanged.

A real guest probe confirmed the form's Name, Reference, Priority, checkbox and
Save button geometry. The updated executor compiled and returned 23 controls
in a 409ms initial observation. A single background MTA worker has a 750ms
caller deadline and an 80-control output limit. Unavailable/busy providers
return no hints; previous observations' hints are never substituted. Edge's
dedicated profile enables renderer accessibility at launch.

The Windows APIs return physical coordinates, matching the DPI-aware executor.
Properties are fetched with a per-observation cache to reduce cross-process
calls. References: [UI Automation screen scaling](https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/ui-automation-and-screen-scaling),
[property caching](https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/caching-in-ui-automation-clients).

Recovery testing also exposed a false indication of progress from a flashing
taskbar icon. The no-progress guard now compares the foreground window's image
and identity instead of the whole desktop. It still retains full PNGs and
their original hashes as evidence. Repeated identical actions against unchanged
foreground pixels pause on the third attempt; the overall 100-tool limit also
remains. This is not a complete semantic loop detector: changing foreground
content or alternating actions can evade the image-based condition.

The corrected clean snapshot is named `qw3-uia-20261001`. The original snapshot
and its failed suite remain intact. A new complete repeated suite is required
before declaring the deterministic tasks accepted.

## Real executor recovery

`windows-recovery-03` passed in 29.13 seconds against the actual guest:

- Three repeated ineffective key actions paused operation; another action was
  rejected while paused.
- A proxy withheld a reply after Windows durably completed an action. Cancelling
  the client retained that action as uncertain and paused the desktop.
- The setup helper killed only the task-owned executor. Its bootstrap restarted
  it, changing PID 8348 to 9092 while preserving the desktop and action ledger.
- Resume queried the old receipt and resolved uncertainty. Repeating the same
  action ID returned the stored receipt. Exactly **one** POST action reached
  Windows; recovery did not replay it.

This is injected transport loss around a real Windows action, not model-driven
fault recovery. `windows-recovery-01` retains a driver error (missing fresh observe
after resume); `windows-recovery-02` retains the taskbar-animation failure.

Separately, `windows-restart-01` passed a real Codex/QW3 cross-application task
after forcibly terminating the host application's own process group following
its first completed action. Restart preserved the task identity and paused
the desktop. No action was dispatched during the two-second paused check.
An explicit resume restored the Codex thread, reconciled the current screen and
finished the copy task. The saved file contained exactly the expected 22-byte
reference code. Total test time was 71.63 seconds, including restart and the
paused interval. This exercised the Linux-host application controlling Windows;
the same process-crash test has not yet run with the app hosted inside Windows.
Evidence: `apps/computer_use/.runtime/windows-restart-01/` and
`windows-restart-engine-01/`.

## Client and lifecycle checks

The private VM model relay passes source-address, token, route and streaming
cancellation checks. The app/bridge accept an optional backend token file;
that token is an HTTP header and does not enter model history, output or request
traces. Missing final desktop evidence now leaves the task paused, even when
the Codex turn reports completion. All **29** local contract/lifecycle tests
pass, including actual pinned Codex with a scripted backend.

Windows x64 client assets and an offline bundle are prepared, with the Codex
package checked against the official release digest. The native Windows bundle
now passes **29/29 tests with exit code 0**, using Windows CPython 3.13.13 and
Codex 0.159.2. This includes real Windows Codex/app-server and stdio MCP with
images, multiple tool rounds and interruption against a scripted backend.
It does not yet prove real-model desktop execution through the Windows client.

Evidence is in `apps/computer_use/.runtime/windows-native-contracts-02/`, including
stderr, the exit-code result, bundle/source hashes and test-helper identity.
The bundle is `windows-native-client-03`, ZIP SHA-256
`d47bf411157d8355b5f3028f93c1aa31ea6425d84edd2906865c9eede8ff2143`.
The earlier wrapper attempt exited without a result; the next run passed every
test but did not preserve the process exit code. Both were retained, and the
wrapper now captures native streams separately and holds the process handle
until it reads the exit code.

The first real-model browser pilot on `qw3-uia-20261001` also passed: 10 actions,
40.01 seconds, with the correct saved JSON. Its checkbox and Save clicks used
the supplied `(198,604)` and `(261,673)` control centers. Evidence is in
`windows-uia-pilot-01/`. A single pilot does not complete the repeated suite.

## Second snapshot suite: all attempts retained, 17/18 passed

`windows-suite-02` completed all 18 planned attempts using the UIA snapshot.
KVMem on passed 9/9; off passed 8/9. The final off cross-app task exceeded the
unchanged 300-second deadline before saving its file. It is a failed trial.
The queued native-client pilot was cancelled by its success gate and never ran.

| Task | On task times (s) | Off task times (s) |
|---|---|---|
| Browser form | 58.60, 39.94, 55.63 | 104.79, 91.74, 92.75 |
| Notepad | 50.79, 40.54, 47.54 | 114.32, 177.17, 195.22 |
| Cross-app copy | 65.72, 56.52, 56.06 | 171.54, 153.95, **300.62 timeout** |

The sampled GPU peaks were 44.03 GiB on and 67.44 GiB off; engine process RSS
high-water was 33.00 and 32.03 GiB respectively. These host counters exclude
the separate client processes. All recorded DFlash summaries reported no
fallback: 17,868/33,565 draft tokens accepted on and 20,260/39,589 off. The latter
includes the failed task's completed inference requests. Largest prompts were
50,876 and 69,419 tokens, below K128K, so this suite does not test sparse
retrieval above budget.

This compares the deployed profiles: on enables prefix reuse, while off does
not. Screenshots, actions and trajectory lengths differ between live trials.
The raw per-trial data in `computer_use_windows_suite_02_20261001.csv` includes
action counts, rejected actions, screenshot-to-action latency, receipt latency,
server visible-output timing and failures. It is not an isolated retrieval
speedup measurement. The frozen-input reports separately disable prefix reuse.

The independent evidence audit passed 17 trial checks and rejected the failed
one, despite the matrix being complete. It verified file contents, sizes and
hashes, task/receipt state, clipboard operations and actual DFlash use. A
separate source audit verified 231 hashes across seven recorded Git commits;
those commits added reports or unused helpers while the tested sources stayed
fixed. Engine and launcher identities were also identical. Evidence is under
`windows-suite-02/{evidence-audit.json,frozen-source-audit.json,trials.csv}`.

### Failure-driven changes awaiting model requalification

The failed trajectory spent about 139 seconds before copying the reference.
It used several positive wheel scrolls while trying to reach lower content and
also sent an out-of-range 10-step scroll. The old tool described positive as up,
but exposed only a signed step count and implicitly used the existing pointer
position. A later Save As transition caused a correctly rejected stale-window
click and another observe/model round. Eventually it typed the filename but
timed out before completing the save.

Scrolling now has an explicit `up`/`down` direction, positive 1–8 steps and
required screenshot coordinates inside the intended pane. The executor moves
the pointer before sending the wheel input. A real probe exposed Edge's smooth
animation continuing through the original 250ms capture delay; scroll now gets
an additional 600ms settling interval. Observation also retries, at most twice,
if the foreground window/title changes while collecting its screenshot and
accessibility metadata. Continued transitions omit control hints rather than
pairing them with a different window's screenshot.

A confirmed pre-dispatch rejection now returns an error **and a fresh screenshot**
to the model. The rejected action remains reserved and is never replayed. Lost
or uncertain delivery still pauses and requires reconciliation. All 31 local
contract tests pass, including pinned Codex preserving images in an MCP error
result and interruption. The latest Windows bundle still needs native execution.

`windows-scroll-contract-03` passed against actual Windows: the Name control
moved from y=335 to y=154 after down, and back to y=335 after up. Wrong direction,
zero/excess steps and out-of-screen coordinates were rejected. Probe 01 retains
a test-driver focus mistake; probe 02 retains the intermediate-animation-frame
failure that motivated the additional settling interval. These are executor
probes, not successful model-driven repetitions.

Real-model requalification, native-client desktop execution, native-client
process-crash recovery, Mac and physical-desktop tests remain open. The task
definitions, 300-second deadline and 100-tool limit remain unchanged.
