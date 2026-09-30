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
result and interruption. Native contract qualification is recorded below.

`windows-scroll-contract-03` passed against actual Windows: the Name control
moved from y=335 to y=154 after down, and back to y=335 after up. Wrong direction,
zero/excess steps and out-of-screen coordinates were rejected. Probe 01 retains
a test-driver focus mistake; probe 02 retains the intermediate-animation-frame
failure that motivated the additional settling interval. These are executor
probes, not successful model-driven repetitions.

The new scroll-compatible client subsequently passed **31/31 native Windows
tests, exit code 0**, including the error-image round trip. Evidence is in
`windows-native-contracts-03/`. Bundle `windows-native-client-04` has ZIP SHA-256
`4cf9d3cf8e5d8cad9c2d5ec4e4e20c7cb24664dc500a2d1fef4d279b8d798945`.
The previous bundle remains archived by its hash. The new powered-off snapshot
`qw3-scroll-20261001` contains this client and executor SHA-256
`2dd1a5ab9a4bad3537515d6a53386c6793107bd9de8cb07058d79e2aca62238a`.
Fixture and bootstrap hashes are unchanged. Native real-model execution is
still pending; these fixture-backed contracts do not qualify that path.

Real-model requalification, native-client desktop execution, native-client
process-crash recovery, Mac and physical-desktop tests remain open. The task
definitions, 300-second deadline and 100-tool limit remain unchanged.

The repaired Linux-host off-mode cross-app pilot (`windows-scroll-pilot-01`)
subsequently passed in **135.96 seconds**, with 13 completed actions and no
rejected action. It used two explicit downward scrolls and real Ctrl+C/Ctrl+V;
the independently read output was the exact expected 22-byte file. The original
300-second timeout remains a retained failure. This one repaired trial does
not replace the required repeated suite. Its success gate has now started the
native Windows browser pilot.

## First native Windows real-model pilot

`windows-native-pilot-01` passed the browser task in **40.66 seconds**, with
10 completed actions, no rejected or unresolved action, and an independently
verified 89-byte JSON file. The app, Codex 0.159.2 and MCP ran inside Windows;
only model inference ran on the local Linux GPU through the private authenticated
relay. The 31 native contracts passed again after snapshot restore. App-tree
shutdown and driver exit codes were both zero, and the guest runtime was exported.

The actual Windows Edge UI screenshot was inspected: Chinese controls rendered,
the final desktop showed the saved form, and the output panel listed the file.
The helper retained 38 valid client process samples including Python and Codex.
Peak summed working sets were 201.16 MiB, peak summed private bytes 110.48 MiB,
and summed sampled CPU time 5.06 seconds. Shared working-set pages can be counted
more than once; these are client-process counters, excluding the controlled
Edge/Notepad processes and the separate GPU server. Sampled host GPU peak was
43.98 GiB. DFlash accepted 1,603/2,790 drafted tokens with no fallback.

The independent native evidence audit passed the trial check. Its overall
matrix result remains false because a pilot contains one of the required 18
trials. This does not yet qualify repeated reliability or native crash recovery.
Evidence: `windows-native-pilot-01/{evidence-audit.json,trials.csv}` and
`on-browser-0/guest/{native-ui.png,client-resources.jsonl,app/,task/}`.

## Native Windows crash/resume

`windows-native-restart-01` passed in **63.12 seconds**, including the injected
crash and pause. The setup driver killed only its own app tree after the first
completed Edge-open action. The app PID changed from 2076 to 9408; the task ID
and Codex thread ID were preserved. The restarted app remained paused, with no
additional action during a two-second hold. Explicit resume completed the
cross-app copy and saved the exact expected 22-byte file. The host independently
confirmed its content/hash and the use of Ctrl+C/Ctrl+V without typing the code.

The guest driver exited zero and both injected-crash and final-cleanup taskkill
commands succeeded. The terminated Python processes themselves returned exit
code 1, as expected for forced termination; this is distinct from the test
driver's successful exit. Evidence is in `windows-native-restart-01/`, including
the exported `guest/restart/result.json`, app ledger and both taskkill logs.
This closes the native process-crash case; repeated core tasks remain open.

## Browser button integration

`browser-ui-check-01` passed in 6.85 seconds using Linux Chromium, Playwright
1.58.0 and actual pinned Codex. All user transitions were driven by the real
page's buttons: start, pause, refresh, resume, then start a second task and stop.
The model and desktop endpoints were scripted local fixtures on independent
ports, so this does not add another Windows/model task success.

The first task preserved its task/thread identities across resume. Both pause
and stop cancelled the held upstream request, and neither dispatched another
action during its two-second hold. The UI showed the delivered file, expanded
its content, and rendered HTML-like model/file text literally without executing
it. There were no page errors. The completed screenshot was visually inspected.
Evidence includes result/source manifests, app/Codex traces and three screenshots
under `apps/computer_use/.runtime/browser-ui-check-01/`. This CPU-only check ran
while the native suite was active; that suite's elapsed times are live workflow
measurements rather than an isolated system-wide latency benchmark.


## Retained native repetition failure and literal-input reproduction

`windows-native-suite-01` stopped at its tenth attempted trial: all nine
KVMem-on tasks passed, then `off-notepad-0` timed out at 300.27 seconds before
saving a file. This is **9/10 attempted, not a completed 18-trial comparison**.
The evidence audit failed the incomplete matrix and the failed trial. The
success-gated live archive queue consequently exited without running a task.

The failed desktop screenshots show actual repeated punctuation and displaced
newlines after a Unicode keyboard burst. The model repeatedly tried to repair
the document; dense re-prefill added delay, but the input corruption was also
reproduced without any model. `windows_typing.py` types through the executor,
uses the visible Save As controls and independently reads each saved file's
UTF-8 content, size and SHA-256. Its initial diagnostic version used only
keyboard events; the final candidate uses the guarded input path described below. The original executor failed all three English trials in
`windows-typing-before-01`; the first file contained `QW3 computer use` followed
by a blank line and repeated periods.

Per-character Unicode pacing did not repair this: the 2 ms attempt
(`windows-typing-after-01`) and 10 ms attempt (`windows-typing-10ms-01`) both
failed their first English file check. The guest executor hash was independently
confirmed for the 2 ms attempt. A current-layout virtual-key attempt
(`windows-typing-native-01`) passed one of four English/Unicode trials; it also
lost uppercase letters and corrupted Unicode. These failed implementations and
raw file/screenshot evidence remain retained. Notepad version was 11.2607.14.0.
No spelling/autocorrect setting was disabled to make a test pass.

Focused-window character messages preserved short text after adding explicit
Enter messages, but synchronous delivery exceeded the action deadline for
4096 units (`windows-typing-wmchar-02`). Asynchronous delivery returned before
Notepad finished processing the input (`windows-typing-postchar-01`). These
remain failures, not qualified input implementations.

## Literal-input repair

The final executor uses synchronous `WM_PASTE` for the focused Windows Edit or
RichEdit control. It snapshots existing clipboard formats before temporary
ownership, prevents the temporary payload from entering Windows clipboard
history/cloud sync, and restores the snapshot on normal completion. Unknown
formats are copied as opaque HGLOBAL bytes rather than deserialized objects;
non-byte formats or individual raw formats above 16 MiB fail before taking
ownership. A newer clipboard change is preserved instead of overwritten.
Browser windows retain Unicode input, now in bounded batches separated by
20 ms, with a focus check and a 14-second dispatch deadline. Partial failures
continue to pause the durable executor and must not be blindly retried.

The initial paste candidate passed **12/12** actual Notepad files across three
repetitions (`windows-typing-paste-01`): English, multilingual/emoji/newlines,
4096 ASCII units and 4096 UTF-16 units containing Chinese/emoji. The final
source, SHA-256 `b5e4eafa2204609a32914ef90bd2aaec85a902884ec7199ea2e3fbb8bddd37fb`,
passed **5/5** in `windows-typing-paste-04`: all four Notepad cases plus a real
Edge form download containing Unicode text and an exact 4096-character field.
Notepad type actions took 0.51–0.59 seconds. Saved/downloaded files, not just
screenshots, supply the independent content/hash checks.

The earlier unpaced browser attempt returned before its subsequent clicks
finished (`windows-typing-paste-02`, four Notepad checks passed; browser gate
failed). A separate probe startup failed because Edge covered the newly opened
Notepad (`windows-typing-paste-03`, no typing dispatched); the probe now uses
ordinary Alt+Tab in that state. Both attempts remain retained.

`windows-clipboard-01/check-final.json` confirms the final guest source and
preservation of all five seeded formats after input: Unicode text, RTF, a
bitmap, a file list and custom bytes. The seed/check helper runs only as VM
setup; the model receives no clipboard-reading tool. Cross-app copy tasks still
require the model's actual Ctrl+C/Ctrl+V sequence. Normal clipboard restoration
is qualified; forced executor termination during temporary clipboard ownership
is not yet qualified. These direct input checks do not replace the repeated
native model tasks, which remain open.

The setup helper archives prior outputs/actions and clears only the dedicated
VM's test clipboard before creating a fresh baseline. The client bundle stays
at version 04 because its app/Codex/MCP files are unchanged.

New clean snapshot `qw3-type-20261001` was created with the final executor hash
above and unchanged fixture/bootstrap hashes. Native and live-archive helpers
now default to this snapshot; older snapshots remain retained.
