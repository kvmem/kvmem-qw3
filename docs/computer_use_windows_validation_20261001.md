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
fault recovery. Application-process crash/resume with real Codex is a separate
pending test. `windows-recovery-01` retains a driver error (missing fresh observe
after resume); `windows-recovery-02` retains the taskbar-animation failure.

## Client and lifecycle checks

The private VM model relay passes source-address, token, route and streaming
cancellation checks. The app/bridge accept an optional backend token file;
that token is an HTTP header and does not enter model history, output or request
traces. Missing final desktop evidence now leaves the task paused, even when
the Codex turn reports completion. All **29** local contract/lifecycle tests
pass, including actual pinned Codex with a scripted backend.

Windows x64 client assets and an offline bundle are prepared, with the Codex
package checked against the official release digest. These artifacts are not
yet Windows-native runtime qualification. The native client, complete live
on/off repetitions, application crash/resume, Mac and physical-desktop tests
remain open.
