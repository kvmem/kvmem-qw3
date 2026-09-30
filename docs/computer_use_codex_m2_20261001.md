# Windows computer-use application: first real tasks

Date: 2026-10-01, Asia/Shanghai. Overall goal remains active.

## Working implementation

`apps/computer_use` now contains a local browser UI, lifecycle controller,
Computer MCP proxy and Windows desktop executor. The controller runs pinned
Codex app-server 0.159.2, routes inference through the local Responses adapter
to QW3, and exposes only `observe` and bounded `act` on the Computer MCP.

The UI displays task progress, current action, the latest desktop screenshot,
pause/resume/stop controls and actual saved-file contents. It uses no external
assets. A Chromium check found no JavaScript errors and confirmed screenshot
loading. The captured UI is retained at `/tmp/qw3-computer-ui.png`.

Each action identifies its desktop and latest observation. The Windows service
checks bounds, age, foreground window and allowed application before dispatch.
SQLite on the host and per-action receipts in the guest retain delivery state.
A duplicate completed ID returns the receipt; an interrupted or uncertain
delivery is never blindly replayed. Resuming first checks the guest receipt,
and unresolved delivery requires explicit user reconciliation. The application
starts paused after an unfinished task is restored.

The executor uses Win32 input and screenshot APIs inside the logged-in guest.
An initial Unicode input smoke exposed dropped line breaks. Sending newline
characters as Enter key events fixed this; saved UTF-8 content with Chinese
text and CRLF line endings was independently verified afterward. The failed
first attempt remains recorded in the runtime action database.

## Environment and actual tasks

- Same RTX PRO 6000 Blackwell Server Edition host, 97,887 MiB reported memory.
- Qwen3.8-27B Q8_0 target, matching vision tower, FP8 KV and DFlash 2.
- Binary SHA-256: `e14b429817f42cedee5659c786087f0265f6335a58073d884a2108782ca9a2ad`.
- Windows 11 Enterprise evaluation, QEMU/KVM, four vCPUs. After a graceful
  restart, QEMU's actual command line confirms `-m 8G`; the earlier installation
  boot had automatically fallen back to roughly 3.6 GiB while caching the ISO.
- Inference and Codex/UI run on Linux; the controlled GUI and executor run in
  Windows. This is not a native Windows client or Windows GPU inference claim.

The three real-model smoke tasks used KVMem K128K/G64K, 1M logical context,
prefix reuse enabled, pin-all visual history, and the initial 32GiB CPU tier.

| Task | Wall time | Completed desktop actions | Independently verified result |
|---|---:|---:|---|
| Create and save a new Notepad document | 38.53 s | 10 | `codex-note.txt`: exact two lines with final CRLF |
| Fill offline Edge form and download JSON | 52.34 s | 20 | `browser-result.json`: Alice / QC-001 / High / confirmed=true |
| Copy from Edge, paste into a new Notepad document, save | 43.89 s | 12 | `copied-reference.txt`: exact `QW3-WINDOWS-2026-ALPHA` |

Task wall time is application start through final observation/file collection.
The cross-app runner measured 44.14 s including its polling. During that task,
pause returned in 104 ms; the desktop remained paused for two seconds, then
the existing Codex thread resumed and finished. The action log contains actual
Ctrl+C and Ctrl+V, rather than typing the expected reference into the editor.

The model made recoverable mistakes: an invalid short action ID, a stale
foreground observation when a save dialog opened, and inaccurate browser
dropdown clicks. The boundary rejected stale/invalid actions; later observations
allowed completion. These mistakes and their latency are included in the results.
The model's completion message was not the verifier: the authenticated guest
file endpoint supplied actual bytes/content, which the runner compared exactly.

There were 54 bridge requests: 53 completed and one cancelled during the pause.
The largest prompt had 33,079 tokens, below the configured retrieval budget.
This proves real image/tool/history handling, not sparse long-history retrieval.

## Reproducible evidence

Ignored runtime artifacts remain local under `apps/computer_use/.runtime/`:

- `windows-app-01/results/{notepad,browser,cross-app}-01.json`: task records and files.
- `windows-app-01/desktop/desktop.sqlite`: requests, receipts and observations.
- `windows-app-01/desktop/screenshots/`: content-addressed desktop PNGs.
- `windows-app-01/codex-events.jsonl` and `codex/`: actual app-server history.
- `windows-app-01/inference/`: exact requests, upstream streams and timings.
- `windows-cross-app-01/result.json`: pause/resume and exact file assertion.
- `dflash-windows-01/`: engine manifest and logs.

Launch and test commands are in `apps/computer_use/README.md`.
`windows_task.py` freezes the three tasks and verifies expected files.
`compare_replay.py` replays hashed, identical requests without dispatching tools.
`vm/snapshot.py` manages only the powered-off, task-owned VM and saves matching
UEFI variable state. Snapshot metadata includes executor/fixture/bootstrap hashes.
The clean snapshot `qw3-clean-20261001` has been created and restored once.
Before creation, outputs and action receipts were archived on the host share,
test applications closed and fixture state reset. Guest inventory at this
baseline reports Windows build 26300 and 8,584,982,528 bytes physical memory.
This snapshot operation does not yet count as repeated task acceptance.

## Checks and remaining acceptance

The application test suite contains 24 passing tests: 11 provider/Codex tests,
seven durable-desktop tests, and six controller/lifecycle/web-origin tests.
The durable-desktop failure tests cover lost responses after an effect,
cancellation after an effect, restart reconciliation, explicit acknowledgement
of an unknown result, duplicate IDs, stale identities and no-progress pausing.
These injected-failure tests do not establish all real Windows failure modes.

The initial tasks were one trial each, on a shared evolving VM state. Still open:
three repeats from a common clean snapshot; real executor crash/app restart;
real no-progress takeover; actual retrieval above K128K; sustained resource and
latency evaluation; native Windows/Mac clients; physical-machine validation.
The VM test account is elevated and its application allowlist is not a complete
OS permission sandbox. Broader deployment needs its own acceptance.

The first comparison attempt retained a native-MTP startup failure: 32GiB CPU
backing was insufficient for its extra KV at 1M context. The common profile was
raised to 48GiB CPU capacity and the complete matrix passed all 54 requests.
See `computer_use_kvmem_comparison_20261001.md` for the results and limits.
A transient busy GPU
lock between profiles was also retained; the runner now waits up to ten seconds
for release and never steals the lock or stops another experiment.
