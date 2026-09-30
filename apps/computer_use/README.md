# Local Codex computer use

Work in progress: the local UI, Codex provider bridge, Computer MCP and Windows
VM executor work end to end. Three real-model Windows smoke tasks have produced
independently verified files, including one pause/resume run. Repeated trials
from a clean snapshot and physical-machine acceptance remain open. A real
Codex observation fixture preserved eight images through 168K tokens and
answered a historical-image question with above-budget KVMem selection.
See `../../docs/computer_use_codex_m2_20261001.md` for desktop evidence and
`../../docs/computer_use_long_history_20261001.md` for the separate diagnostics.
See `../../docs/computer_use_codex_goal_20260930.md` for the full goal.

## Runtime

The intended application uses the pinned **Codex CLI/app-server 0.159.2** as its
agent harness. `qw3_computer.codex` starts it over JSONL stdio, in an app-owned
`CODEX_HOME`, with a local model catalog and provider. It does not load the
user's cloud credentials, MCP connections or plugins. Shell, built-in browser,
image generation and multi-agent features are disabled. Only the dedicated
Computer MCP's `observe` and `act` tools are enabled on that server.

Codex sends Responses requests to `qw3_computer.bridge`; the bridge streams
Chat Completions from the existing QW3 server. All inference, image input and
history stay on the local machines. The bridge accepts one inference request
at a time. Client cancellation closes the upstream connection, including
during silent prefill. No retry is performed after an uncertain tool delivery.

The current protocol subset supports text, inline PNG/JPEG/WebP images,
function schemas, namespace mapping, tool outputs containing images, full
history replay and Responses SSE events. It rejects stored/previous-response
state, remote/file-ID images, custom tools, strict schema enforcement, reasoning
efforts other than `none` and unknown input item types. It is **not** a general
Responses API implementation. Native QW3 thinking is disabled in this initial
profile; DFlash remains active. Tool calls are validated and released only
after the backend's complete terminal event. Partial or malformed tool calls
are never delivered for execution.

The model catalog advertises image input and a 1M logical context. Automatic
compaction starts at 950K tokens, above the 128K KVMem retrieval threshold.
Image-budget compaction is disabled. Preservation and retrieval passed a 168K
diagnostic; Codex's behavior near the 1M context limit remains unqualified.

## Development and tests

Use Python 3.11+ with the dependencies in `pyproject.toml`. On the current
development host, the repository's `.venv/bin/python` already has them.
Commands below run from the repository root:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python -m unittest discover \
  -s apps/computer_use/tests -v

# Includes real Codex + real stdio MCP against a scripted local backend.
# Requires local socket access; does not use a cloud API or GPU.
PYTHONPATH=apps/computer_use QW3_TEST_CODEX=1 .venv/bin/python \
  -m unittest discover -s apps/computer_use/tests -v
```

The integration fixture returns generated color images; it never controls a
desktop. The tests verify image preservation, tool IDs, stable multi-round
history, actual Codex turn completion, interrupt propagation, stream failure,
truncation and malformed requests. Fixture tests are not desktop-task results.

## GPU smoke

Build `build-dflash-integration` first using the CUDA/FlashInfer configuration
documented in `../../docs/dflash2_integration.md`. Start one engine, using a new
run directory for each attempt:

```bash
python3 apps/computer_use/run_engine.py \
  --method dflash2 --run-dir apps/computer_use/.runtime/engine-new-run

# In another terminal, once engine.log says it is listening:
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/smoke_codex_gpu.py \
  --run-dir apps/computer_use/.runtime/codex-new-run
```

The launcher records binary hash, source commit, GPU, exact arguments and
QW3 environment. It checks for existing GPU compute processes and holds the
shared DFlash experiment lock when present. It requires local model assets:

- `models/Qwen3.8-27B-Q8_0.gguf`
- `models/Qwen3.8-27B-DFlash2-BF16`
- `models/Qwen3.8-27B-BF16` for the matching visual tower

The existing binary has CUDA tensor-allocation and workspace-capacity tracing
hooks. `run_engine.py --trace-memory` enables them; `compare_replay.py` accepts
the same flag for a separate paired diagnostic. Do not use instrumented timings
as a latency benchmark. Tensor records cover owned allocations of at least 1 MiB; workspace
counters cover named backend capacities. Views, smaller/direct allocations and
all CUDA-library internals are not a complete census.

`summarize_memory.py` collects retained component/capacity logs and separate
resource samples. It explicitly marks an ordinary run's missing allocation
traces; `--require-traces` rejects a missing or inconsistent diagnostic. For
example, after an engine has exited:

```bash
python apps/computer_use/summarize_memory.py /path/to/run/engine \
  --samples /path/to/run/resources.jsonl --output /path/to/run/memory-evidence.json
```

Its allocation-event arithmetic was checked with synthetic records, including
unmatched frees and malformed input. The retained long DFlash log reproduces
the known resource peak and reports that allocation tracing was absent. An
instrumented DFlash long-context on/off run has now passed all four visual
answer checks and both trace-consistency checks; see
`docs/computer_use_memory_diagnostic_20261001.csv`. The counters remain a partial
allocation view, not a full allocator census.

The profile uses FP8 KV, K128K/G64K, 1M logical context, a 48GiB CPU
tier, 128-token blocks with four 32-token scoring slices, message-level
retrieval and multimodal prefix reuse. It uses the available 96GB-class GPU;
48GB is not an acceptance limit. `--method none` and `--method mtp` select
separate controls. `--kvmem on|off` selects the memory policy, and
`--prefix-reuse off` disables prefix reuse for paired fixed-input replay.
The first DFlash smoke used a 32GiB CPU tier. A later MTP startup check showed
that its additional KV backing exceeds that budget at 1M context, so the common
comparison profile uses 48GiB. This is CPU memory capacity, not a GPU limit.

The smoke checks a real model identifying a red input image, requesting two
MCP fixture observations and identifying the returned blue screenshots. It
retains the Codex events and its success/failure result. This does not measure
Windows task completion or sustained performance.

`smoke_ui.py` exercises the actual browser buttons with real pinned Codex,
the app, bridge and MCP, using scripted local model/desktop endpoints on
independent ports. With Playwright 1.58.0 and an existing Chromium executable:

```bash
PYTHONPATH=apps/computer_use python apps/computer_use/smoke_ui.py \
  --browser /absolute/path/to/chrome \
  --run-dir apps/computer_use/.runtime/browser-ui-new
```

`browser-ui-check-01` passed start, pause, refresh, same-task/thread resume,
file expansion, and a second task's stop. Both held inference requests were
cancelled; no extra action occurred during two-second paused/stopped checks.
File and model text containing HTML rendered literally, with no browser errors.
It retained screenshots of paused/completed/stopped UI and the Codex trace.
This is a Linux browser wiring check with fixture services, separate from
Windows real-model desktop acceptance. Playwright is a test-only dependency.

## Windows VM

`vm/start_windows.py` creates only the task-owned `qw3-computer-windows`
container, with a digest-pinned Dockur/QEMU runtime, KVM, Windows 11 Enterprise
evaluation media, a dedicated test account and a sparse 64GiB virtual disk.
Existing containers are never replaced. Runtime files, credentials, image
downloads and VM disks live under `.runtime/windows` and are ignored by Git.
The launcher explicitly clears lowercase and uppercase HTTP/HTTPS/ALL proxy
variables for the Windows ISO downloader, following the user's preference.

```bash
python3 apps/computer_use/vm/start_windows.py
docker logs --tail 30 qw3-computer-windows
```

Only loopback ports are published: browser viewer `http://127.0.0.1:18006`,
VNC `15900`, and the desktop executor `18082`. The VM has no GPU
passthrough. The GPU model remains on the Linux host.

The launcher stages `executor.ps1`, the offline fixture and a random token into
the VM's dedicated share. The bootstrap runs them in its logged-in test account
desktop and configures this account's Edge download folder. It never runs on
the host. The executor exposes screenshots, bounded click/type/key/scroll and
three allowed applications (Notepad, Edge, Explorer). Every action refers to
the current desktop and observation, expires after 60 seconds, and checks the
foreground window. Completed action IDs return their durable receipt. An
uncertain delivery pauses operation and must be reconciled before continuing.

Start the engine above, then launch the application:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python -m qw3_computer.app \
  --runtime apps/computer_use/.runtime/my-desktop-run \
  --desktop-token apps/computer_use/.runtime/windows/shared/executor-token.txt
```

Open `http://127.0.0.1:18083`. The UI shows progress, latest screenshot, pause,
resume/stop controls and saved-file contents. The viewer on `18006` supports
manual takeover while the application is paused. Refreshing the UI image does
not invalidate the model's active observation. SQLite action records, images,
Codex history and exact inference requests stay in the app-owned runtime.
On restart, unfinished tasks are paused until explicitly resumed.

```bash
# Start each trial with no pre-existing expected output file.
.venv/bin/python apps/computer_use/windows_task.py --case cross_app \
  --pause-after-seconds 6 --run-dir apps/computer_use/.runtime/cross-app-new

# Stop the application first; keep the VM running for fixture preparation.
python3 apps/computer_use/vm/provision.py apps/computer_use/vm/prepare_baseline.ps1
# Snapshot operations require the VM powered off.
docker stop -t 120 qw3-computer-windows
python3 apps/computer_use/vm/snapshot.py create --name clean-baseline
docker start qw3-computer-windows
# Before a repeat: stop, restore the same named snapshot, then start again.
```

`windows_task.py` freezes three task definitions and checks actual saved text
or JSON. A completed model turn alone never qualifies the file check. Initial
smoke tasks were not repeated from a common snapshot; they are not a benchmark
success rate. `vm/provision.py` is a setup-only QEMU console helper, never a
model-accessible tool. `vm/vnc_admin.py` captures diagnostic screenshots only.

`windows_suite.py` automates the same-snapshot repetitions. Stop the app and
engine first; the suite restores only the task-owned VM and starts a fresh app
and engine for every trial. It checks saved files directly through the guest
and checks Ctrl+C/Ctrl+V action receipts for the copy task. Every failure and
source identity is retained. A completed suite can still contain failed trials;
inspect `summary.json` rather than treating process exit as task success.

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_suite.py \
  --snapshot qw3-scroll-20261001 --kvmem on off --repetitions 3 \
  --run-dir apps/computer_use/.runtime/windows-suite-new
```

The Windows guest currently runs the executor; the UI, Codex and GPU model run
on Linux. A packaged native Windows/Mac application and physical-desktop tests
are separate qualification steps. The dedicated VM test account is elevated;
this prototype is not qualified as a general-purpose desktop permission sandbox.

The second full on/off snapshot suite retained 17 successes and one off-mode
cross-app timeout. The current revision addresses its scroll/navigation failure:
`scroll` requires `direction` (`up`/`down`), positive `steps` (1–8), and `x`/`y`
inside the intended content pane. Old signed-only scroll arguments are rejected.
The executor waits briefly for wheel animation and rechecks foreground changes
during observation. Confirmed rejected actions return a fresh screenshot without
replaying input. The 300-second task deadline and 100-tool limit are unchanged.
These changes passed a real Windows scroll probe and all 31 contracts on both
Linux and Windows. The matching `qw3-scroll-20261001` snapshot and
`windows-native-client-04` bundle are prepared; model-driven requalification
remains pending. Earlier snapshots retain their original source hash guards.

### Recovery and native-client qualification helpers

After the repeated suite has stopped, `windows_recovery.py` exercises the real
executor's repeated-action pause, then drops one reply after Windows durably
completes an action. It checks reconciliation and duplicate receipts without a
second dispatch. `--restart-executor` additionally restarts only the dedicated
guest executor before reconciliation. `windows_restart.py` requires a clean
snapshot and a running engine; it crashes its own application process group,
checks that restart stays paused, explicitly resumes, and verifies the task file.
The real executor test passed, including an executor restart and exactly one
action dispatch across lost-reply recovery. A real Codex/QW3 task also passed
application crash/resume with the app on Linux and executor on Windows, retaining
pause until explicit resume and independently verifying the copied file. See
`../../docs/computer_use_windows_validation_20261001.md` for retained failures
and the corrected foreground-window progress check.

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_recovery.py \
  --restart-executor --run-dir apps/computer_use/.runtime/recovery-new

PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_restart.py \
  --run-dir apps/computer_use/.runtime/restart-new

python3 apps/computer_use/summarize_windows.py apps/computer_use/.runtime/windows-suite-new

PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/audit_windows_suite.py \
  apps/computer_use/.runtime/windows-suite-new
```

The evidence audit requires all 18 terminal trials and checks exact saved
contents and hashes, task identities, final pause, clipboard actions, unresolved
receipts, model configuration and active DFlash counters. It correctly rejects
the retained incomplete first suite while validating its seven successful trials.
Different recorded Git commits can reflect documentation or new unused helpers;
the engine binary and launcher hashes must still be identical across the suite.

`pack_windows_client.py` assembles an offline x64 client folder from CPython
3.13.13, the official Codex 0.159.2 package and Windows wheels. It verifies Codex
against the official release digest and records every asset/source hash. The
folder includes `start.ps1` and `test.ps1`; it contains no model weights or
executor token. `vm/test_native_client.ps1` is a setup-only helper to run the
contract tests in the dedicated guest from a staged, hashed ZIP. The current
Windows bundle 04 passes all 31 tests with exit code 0, including actual native
Codex and MCP success/error image results against a fixture backend. A native
Windows real-model browser pilot also passed in 40.66 seconds with 10 actions
and independently verified JSON. Full repeated acceptance remains open.

`vm/model_relay.py` connects the native Windows client with
the host GPU. It binds only the owned container's Docker bridge gateway and
requires both that container's source address and a separate token. Its route,
authorization and disconnect behavior pass local tests. The client accepts
`--backend-token` (or `-BackendToken` in its PowerShell launcher) as a file path;
the token stays outside model history. Real VM connectivity passed in that pilot. The runtime bridge
and model stay on loopback in the existing Linux-host trials.

`windows_native_suite.py` and the setup-only `vm/run_native_case.ps1` prepare
real-model qualification with the app, native Codex and MCP inside Windows.
Each trial restores the clean snapshot, installs and tests the hashed offline
client, starts the private model relay, then checks the saved task file from
both the native client and the host. It exports the guest runtime and renders
the actual local UI in Windows Edge. Their first browser pilot passed, including
process-tree cleanup and evidence export. Wait for other VM/model suites to end
before running them; restoring a snapshot resets the dedicated guest.
The native task helper also samples its app process tree's CPU time, working
set and private-byte counters. Shared pages can appear in multiple working
sets, so their sum is not unique physical RAM use. These guest counters are
separate from the host GPU/engine resource log. The pilot retained 38 valid
samples covering Python and Codex. Its peak sum of working sets was 201.16 MiB
and peak sum of private bytes was 110.48 MiB; these exclude the GPU engine and
the controlled desktop applications.

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_native_suite.py \
  --cases browser --run-dir apps/computer_use/.runtime/windows-native-pilot-new

PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_native_suite.py \
  --restart --cases cross_app --run-dir apps/computer_use/.runtime/windows-native-restart-new

PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_native_suite.py \
  --kvmem on off --repetitions 3 --run-dir apps/computer_use/.runtime/windows-native-suite-new
```

`summarize_windows.py` and `audit_windows_suite.py` also read native suite
exports. The audit still requires all 18 repeated tasks; a passing single pilot
is reported as an incomplete matrix. Native checks additionally verify bundle
and setup-helper hashes, process cleanup, client samples and UI output.

The restart variant kills only the application process tree it started, checks
that restart stays paused with the same task identity, then explicitly resumes
and verifies the saved file. `windows-native-restart-01` passed in 63.12 seconds:
the same Codex thread resumed after a two-second no-action pause and produced
the exact expected cross-app file using Ctrl+C/Ctrl+V. That time includes the
injected crash and pause, but excludes VM/model setup.

## KVMem and decoder comparison

Freeze `inference/*/request.json` inputs into a corpus with a `manifest.json`
containing each case's `name`, relative `path`, and `sha256`. Use identical
payloads, image bytes, generation limits and order for both memory policies.
Stop the application and its engine before running the comparison:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/compare_replay.py \
  --corpus apps/computer_use/.runtime/replay-corpus-01 \
  --run-dir apps/computer_use/.runtime/comparison-new
```

This runs KVMem on/off × none/MTP/DFlash, three repetitions by default, with
prefix reuse disabled. It records exact launch manifests, streamed outputs,
latencies, CPU time, process RSS and sampled GPU resources. Vision cache starts
fresh for each profile and remains warm across its ordered requests; distinguish
the first pass from subsequent passes. It never dispatches generated actions.
Startup/capacity failures are retained without silently truncating one side.
Short histories below K128K qualify overhead and function, not retrieval benefits.

For above-budget diagnostics, start a fresh engine with
`--visual-history retrievable --trace-retrieval`. This allows old visual spans
to participate in retrieval. The default `pin-all` policy protects all images
and cannot fit an unlimited screenshot history into the fixed K budget.

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/long_history_probe.py \
  --engine-dir apps/computer_use/.runtime/engine-new-run \
  --run-dir apps/computer_use/.runtime/history-new-run

PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/smoke_codex_long.py \
  --engine-dir apps/computer_use/.runtime/engine-new-run \
  --historical apps/computer_use/.runtime/history-new-run/historical.png \
  --current apps/computer_use/.runtime/history-new-run/current.png \
  --run-dir apps/computer_use/.runtime/codex-history-new-run
```

The first diagnostic uses recorded screenshots from the frozen short corpus
and synthetic archive notes. It checks short recognition, budget crossing,
retrieval and visual-cache invalidation after replacing the historical image.
The second uses real Codex and an observation fixture; it never sends desktop
input. Use a fresh engine for each isolated measurement, and distinguish warm
prefix reuse from cold on/off comparisons. The long cold corpus can be replayed
with `compare_replay.py --visual-history retrievable --trace-retrieval`.

`windows_long_task.py` prepares a separate real-desktop long-history check:
36 local archive pages, a checkbox and confirmation on each, followed by typing
the opening page's reference into the final form and saving the JSON. It must
cross K128K at the reference-generating request, retain the historical images,
and have no reference text in earlier model history. It checks the final file,
clipboard/page-reset shortcuts and the actual final retrieval trace. The
application's 100-tool limit is unchanged. This workload has not yet passed a
real-model trial and is not included in the three frozen short tasks.
The page itself passed a Linux headless-browser check: all 36 confirmations,
required checkbox behavior, reference hiding after page 1, exact JSON download
and zero JavaScript errors. The first/final layouts were also visually checked
at 1280×675. Evidence is in `.runtime/windows-long-fixture-check-01/`; this
checks the test fixture, not Windows or the model's ability to complete it.

After other VM/model tests end, restore the clean snapshot, copy
`vm/long_fixture.html` into the VM's shared directory, archive any old
`long-task-ready.json`, then run `vm/prepare_long_task.ps1` using `vm/provision.py`.
Wait for the new ready file and verify its fixture hash. Start a fresh engine
with `--visual-history retrievable --trace-retrieval` and a fresh app runtime,
then run:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_long_task.py \
  --app-runtime apps/computer_use/.runtime/archive-app-new \
  --engine-dir apps/computer_use/.runtime/archive-engine-new \
  --run-dir apps/computer_use/.runtime/archive-task-new
```

`windows_long_suite.py` automates that clean-snapshot setup, fixture hash check,
fresh engine/app startup, resource sampling and owned-process cleanup. It runs
the client on Linux and the desktop on Windows, and refuses occupied service
ports. The orchestrator is prepared but has not yet completed a live trial:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_long_suite.py \
  --kvmem on --run-dir apps/computer_use/.runtime/archive-suite-new
```

Use `--kvmem off` in a separate new directory for the paired live policy. Keep
the same task deadline (900 seconds by default); retain a timeout or a task
that finishes below the retrieval threshold as failed qualification.

After a long replay finishes, validate both the visual answers and the complete
final retrieval trace. This checker has passed all 12 positive/negative cases
in the retained target-only, native MTP and DFlash on/off runs:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/check_long_replay.py \
  apps/computer_use/.runtime/comparison-long-new
```

Sources: [Codex app-server](https://learn.chatgpt.com/docs/app-server),
[Codex config](https://learn.chatgpt.com/docs/config-file/config-reference),
[Dockur Windows](https://github.com/dockur/windows),
[Microsoft evaluation media](https://www.microsoft.com/en-us/evalcenter/download-windows-11-enterprise).


## Windows literal input qualification

`windows_typing.py` verifies actual Notepad saved files for English,
multilingual/emoji/newline text and two 4096-UTF-16-unit boundaries. Add `--browser`
for an independent Edge form download containing Unicode and a 4096-character
field. It uses ordinary desktop actions and Save As, then checks file content,
size and hashes. Run it only while other VM suites are stopped:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_typing.py \
  --run-dir apps/computer_use/.runtime/typing-new --repetitions 3 --browser
```

Edit/RichEdit input uses a synchronous paste message with temporary clipboard
ownership and restoration; other windows use paced Unicode keyboard batches.
Raw clipboard formats are bounded to 16 MiB each and are never deserialized.
Unsupported snapshot formats reject before taking ownership. The final direct
input candidate passed four Notepad cases, the Edge download and preservation
of five clipboard fixture formats. Forced executor termination during temporary
clipboard ownership remains unqualified. See the Windows validation report for
retained failures and the exact source/evidence hashes.
