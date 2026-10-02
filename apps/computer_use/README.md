# Local Codex computer use

The current product goal is tracked in the
[delivery plan](../../docs/computer_use_product_delivery_20261002.md).
The latest [Windows focus repair](../../docs/computer_use_focus_improvement_20261003.md)
passed three fixed switching trials (baseline: zero of three). Native client
21 passed 104 runtime contracts and includes an opt-in controlled Python setup
workflow. These runtime checks are separate from product acceptance; the
[software trial report](../../docs/computer_use_python_workflow_20261003.md)
records three development attempts, including the first successful cold install.
LaTeX and optional advisor integration remain open.

Work in progress: the local UI, Codex provider bridge, Computer MCP and Windows
VM executor work end to end. A frozen native Windows client passed 18 core-task
trials from a clean snapshot. The repaired Linux client completed a 36-page
Windows task with KVMem on; off timed out at the same 900-second deadline.
The current application bundle 06 passed 18/18 native tasks and both on/off
restart checks. Fixed-input controls passed 54 short and 12 long checks across
target-only, native MTP and DFlash. Corrected client resource measurements passed
a separate on/off Notepad pair. The checkpoint boundary repair also passed a
matched 50-request replay, eight visual checks and a new live archive comparison:
on finished in 416.09s without checkpoint misses or rejected actions; off timed
out at page 18. New-engine fixed controls passed 54/54 short and 12/12 long
checks. These results retain separate engine identities; native Windows core
trials used the preceding engine. Physical-machine and macOS acceptance remain
open. See the current
[acceptance index](../../docs/computer_use_acceptance_20261001.md) for exact
source identities, preserved failures and the scope of each result.
See `../../docs/computer_use_codex_m2_20261001.md` for desktop evidence and
`../../docs/computer_use_long_history_20261001.md` for the separate diagnostics.
See `../../docs/computer_use_codex_goal_20260930.md` for the full goal.
The new [ten-scenario Windows campaign](../../docs/computer_use_windows_execution_optimization_20261001.md)
is in platform preparation. Bundle 07 passed 40 native client contract tests;
it has not yet run the ten OSWorld-derived scenarios. This does not replace the
frozen bundle 06 task results above.

## Runtime

The intended application uses the pinned **Codex CLI/app-server 0.159.2** as its
agent harness. `qw3_computer.codex` starts it over JSONL stdio, in an app-owned
`CODEX_HOME`, with a local model catalog and provider. It does not load the
user's cloud credentials, MCP connections or plugins. Shell, built-in browser,
image generation and multi-agent features are disabled. Only the dedicated
Computer MCP's `observe` and `act` tools are enabled by default. Explicit native
Windows software mode also enables the reviewed `software` tool described below.

Codex sends Responses requests to `qw3_computer.bridge`; the bridge streams
Chat Completions from the existing QW3 server. All inference, image input and
history stay on the local machines. The bridge accepts one inference request
at a time. Client cancellation closes the upstream connection, including
during silent prefill. No retry is performed after an uncertain tool delivery.

The current protocol subset supports text, inline PNG/JPEG/WebP images,
function schemas, namespace mapping, tool outputs containing images, full
history replay and Responses SSE events. It rejects stored/previous-response
state, remote/file-ID images, custom tools, strict schema enforcement, reasoning
efforts other than `none` in its default profile and unknown input item types.
It is **not** a general Responses API implementation. Native QW3 thinking is
disabled by default; DFlash remains active. Tool calls are validated and released only
after the backend's complete terminal event. Partial or malformed tool calls
are never delivered for execution.

The application now accepts an explicit `--thinking` candidate option. It selects
local QW3 thinking via the bridge's opt-in `medium` mode; this is a mode switch,
not an implementation of OpenAI effort budgets. Raw local reasoning travels in
separate `reasoning` items and is replayed across tool/user turns. The same output
token cap includes reasoning. Resume/follow-up requires the task's saved mode.
Default launches retain the original non-thinking behavior. This candidate passed
real local-model image/tool checks and native Windows/Codex product tasks, with
actual reasoning-stream and DFlash evidence. See the
[native thinking report](../../docs/computer_use_native_thinking_20261003.md) and
[LaTeX product report](../../docs/computer_use_latex_product_20261003.md) for the tested scope and limits.

The current [product delivery goal](../../docs/computer_use_product_delivery_20261002.md)
prioritizes complete Python/LaTeX software workflows, using diagnostic results to
improve execution, verification and recovery. Optional DeepSeek advisory support
is planned after local baselines; it is not enabled or implemented yet.

## Controlled Windows software mode

The current reviewed profile, `python-web`, installs CPython 3.13.16 x64 in the
selected workspace and creates `projects/<name>/.venv` with requests 2.34.2,
pytest 9.1.1 and their exact wheel dependencies. `software_catalog.json` freezes
official download URLs and SHA-256 checksums. Downloads bypass environment
proxies and use the client's pinned public CA bundle; certificate and hostname
verification stay enabled. Package installation is offline from the verified
wheel cache, using pip hash checking.

Run inside Windows with its already configured local executor and model endpoint:

```powershell
.\start.ps1 -SoftwareRoot 'C:\Qw3Computer\software' -OutputPolicy task
```

In the application, request: “为 python-demo 配置 Python 3.13.16，使用 python-web
配置档，创建隔离环境，完成后核对依赖并说明如何打开项目终端。” Existing source files
can be placed in `C:\Qw3Computer\software\projects\python-demo` before the task.
Project names are short lowercase names, not arbitrary paths. The result includes
`requirements.lock`, `environment-verification.json` and `open-terminal.cmd`;
open the latter to activate that project's environment in a new terminal.

The model selects the profile and checks the job; the application produces the
installation commands. It does not accept model-supplied shell commands, URLs
or packages. System PATH and unrelated Python installations are preserved. A
conflicting Python 3.13 registration or unmanaged existing `.venv` stops setup
for inspection. `repair_python` first diagnoses a matching managed project,
checks its actual imports as well as package metadata, reinstalls the exact
reviewed wheels when necessary, and repairs the delivered terminal launcher
with a saved backup. It verifies the selected interpreter and imports in two
new processes. It preserves source files and does not reinstall system Python.
Unmanaged environments, unknown interpreter identities and changed dependency
locks still require inspection; this is not a general Python repair tool.

For an existing managed project, request: “修复 repair-target 项目，先诊断运行和
终端环境的问题，保留源文件及其他项目，验证后说明修复内容。” Add `-Thinking`
to the Windows launcher to use the locally qualified reasoning mode.

Jobs retain stage results, errors, logs and action identity. Pause cancels known
owned work; downloads and known-stopped jobs can resume under the same identity.
An interrupted installer or application restart during installation is marked
uncertain and blocks another dispatch. Killing an installer does not establish
rollback; reconciliation for that case remains a product limitation. Model
completion alone cannot mark an unverified environment successful.

The owned-VM harness restores the frozen clean baseline, uses a qualified native
bundle, records continuous video, then independently runs the supplied project,
pytest and pip checks from new activated command prompts. Each attempt needs a
new output directory; all failures remain in the report. This is a local-model
workflow with controlled software tools, not a pure GUI benchmark.

```bash
# Start a separate, owned baseline engine first.
.venv/bin/python apps/computer_use/run_engine.py --method dflash2 --kvmem off \
  --run-dir apps/computer_use/.runtime/product-python-engine-next

# Once that engine is ready, run this in another terminal.
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/software_product_run.py \
  --output apps/computer_use/.runtime/product-python-new-next \
  --bundle apps/computer_use/.runtime/windows-native-client-21 \
  --engine apps/computer_use/.runtime/product-python-engine-next \
  --ffmpeg /absolute/path/to/ffmpeg
```

The engine must be running and owned by its manifest; the harness requires the
task-owned VM to be paused before restoring its snapshot. For repair development,
`--repair-fixture <result.json> --thinking` preserves an independently prepared
broken-project VM instead of restoring it. The guest verifies the faults before
starting the model, and checks source hashes, unrelated project contents, PATH,
fresh processes, the actual launcher and application reopening afterward.
Repair and LaTeX acceptance are not implied by a successful new-environment run.

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
  --method dflash2 --kvmem on --guided-refresh both --visual-history retrievable \
  --run-dir apps/computer_use/.runtime/engine-new-run

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

For long tool histories within one user task, `--guided-refresh both` enables
private query generation and retrieval refresh at request/generation boundaries.
It requires DFlash or MTP on the KVMem side; the dense side ignores this option.
The default remains `off` for reproducing earlier measurements. With refresh
off, same-query history growth can use capacity selection without semantic
retrieval. A 162K-token diagnostic exposed an incorrect old-image answer in
that configuration. With refresh on, all four long on/off positive/changed-image
negative checks passed, each seeded by the same short request. That diagnostic
does not dispatch desktop actions. The separate live archive pair and its source
identities are linked in the acceptance index.

```bash
python3 apps/computer_use/run_engine.py --method dflash2 --kvmem on \
  --guided-refresh both --visual-history retrievable --trace-retrieval \
  --run-dir apps/computer_use/.runtime/engine-guided-new-run
```

`windows_long_suite.py --guided-refresh both` selects the same refresh policy
for a new live qualification run. Use new output directories and retain prior
failures. The underlying engine binary is unchanged by this launcher option.

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

For longer campaigns, `--tool-call-limit 500` sets the lifetime observe/act
budget of each **new** task. Failed/rejected calls also consume a slot. The
saved count and limit survive pause/resume and application restarts; changing
the launch option never raises an existing task's budget. An unfinished task
from a legacy client with no persisted count must be stopped before starting
a new one. `--output-policy task` omits the legacy `C:\Qw3Computer\outputs`
save hint, allowing the task's explicit in-place/output paths to govern.
Windows `start.ps1` accepts `-ToolCallLimit` and `-OutputPolicy` for these options.

The host can independently record the shared QEMU console without issuing
executor observations or input:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python -m vm.record_console \
  --ffmpeg /absolute/path/to/ffmpeg --fps 10 \
  --output apps/computer_use/.runtime/new-recording
```

SIGTERM/SIGINT finalizes the recording; `--duration 10` supports a bounded
probe. Keep the recovery MKV, browser-compatible MP4, capture event log and
manifest. Video time uses the host monotonic clock; delayed captures retain
the preceding frame and all actual capture times remain available. Encoded
frame rate is not a claim of constant capture frequency. Recording failure is
explicit in the manifest. This Linux-host helper is separate from the native
Windows agent client. Dynamic-screen decoding and preservation of the active
executor observation passed a live probe; full-campaign recording overhead
and long-duration coverage remain to be qualified.

```bash
PYTHONPATH=apps/computer_use .venv/bin/python -m unittest vm.test_record_console -v
```

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
  --snapshot qw3-edit-20261001 --kvmem on off --repetitions 3 \
  --guided-refresh both --visual-history retrievable \
  --run-dir apps/computer_use/.runtime/windows-suite-new
```

In `windows_suite.py`, the Windows guest runs the executor while the UI, Codex
and GPU model run on Linux. `windows_native_suite.py` runs the client and Codex
inside Windows; bundle 06 passed its 18 repeated core tasks on engine
`005654a6...`. Revised bundles/engines, macOS and physical desktops retain their
own qualification scope. The dedicated VM test account is elevated;
this prototype is not qualified as a general-purpose desktop permission sandbox.

The second full on/off snapshot suite retained 17 successes and one off-mode
cross-app timeout. The current revision addresses its scroll/navigation failure:
`scroll` requires `direction` (`up`/`down`), positive `steps` (1–8), and `x`/`y`
inside the intended content pane. Old signed-only scroll arguments are rejected.
The executor waits briefly for wheel animation and rechecks foreground changes
during observation. Confirmed rejected actions return a fresh screenshot without
replaying input. The 300-second task deadline and 100-tool limit are unchanged.
These changes first passed a real Windows scroll probe and 31 contracts on both
Linux and Windows. Subsequent input repairs and bundle 06 passed 34 contracts
and the 18-task native matrix on `qw3-exact-20261001`, as recorded in the
acceptance index. Earlier snapshots retain their original source hash guards.

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
Windows bundle 06 passes all 34 tests with exit code 0, including actual native
Codex and MCP success/error image results against a fixture backend. The native
real-model core matrix also passed all 18 repeated tasks. Source identities and
every repetition are recorded in the acceptance index.

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
separate from the host GPU/engine resource log. The original PID-only sampler
could attach old system processes after PID reuse, so legacy client counters
are withheld. The repaired observer checks each PID and process creation time,
retains parent identities, and refuses partial/ambiguous samples. Its ten real
Windows probe checks and eight offline validator tests passed. A fresh matched
Notepad pair then passed with all 89 samples qualified: peak summed working
sets were 210.72/214.74 MiB on/off, private bytes 122.63/126.17 MiB. This is one
resource trial per policy, excluding the model and controlled desktop apps. It
does not replace the existing 18 functional outcomes or repair legacy counters.

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_native_suite.py \
  --cases browser --guided-refresh both --visual-history retrievable \
  --run-dir apps/computer_use/.runtime/windows-native-pilot-new

PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_native_suite.py \
  --restart --cases cross_app --guided-refresh both --visual-history retrievable \
  --run-dir apps/computer_use/.runtime/windows-native-restart-new

PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_native_suite.py \
  --kvmem on off --repetitions 3 --guided-refresh both --visual-history retrievable \
  --run-dir apps/computer_use/.runtime/windows-native-suite-new
```

Current native/long suite defaults target bundle 06 and `qw3-exact-20261001`.
Create and verify that matching baseline before running; the older bundle 04
and `qw3-type-20261001` snapshot belong to the clipboard-preserving executor.
The suite records the requested refresh and visual-history settings, and its
audit checks those settings against each actual engine manifest.

`summarize_windows.py` and `audit_windows_suite.py` also read native suite
exports. The default audit requires all 18 repeated tasks; a passing single pilot
is reported as an incomplete matrix. An explicit `--cases notepad --repetitions 1`
audits a separate two-policy resource pair without claiming the full matrix.
Native checks additionally verify bundle
and setup-helper hashes, process cleanup, client samples and UI output.
Client-resource qualification is reported separately from functional success.

After all existing GPU/VM queues end, the setup-only ownership probe can be
repeated against the observer embedded in the checked-in native task helper:

```bash
.venv/bin/python scripts/computer_use_resource_probe.py \
  --after-unit NAME-OF-COMPLETED-TEST.service \
  --run-dir apps/computer_use/.runtime/native-resource-probe-new
```

The probe checks the owned, paused Windows VM and creation identities. It does
not measure the application's memory. Its output must pass before a resource
trial; invalid or legacy records never produce published client RAM totals.

The restart variant kills only the application process tree it started, checks
that restart stays paused with the same task identity, then explicitly resumes
and verifies the saved file. `windows-native-restart-02` passed on/off in
63.03/124.67 seconds on engine `005654a6...`: the same Codex thread resumed
after a two-second no-action pause and produced the exact expected cross-app
file using Ctrl+C/Ctrl+V. Those times include the injected crash and pause,
but exclude VM/model setup. The retained supervisor audit-key error and its
independent evidence review are documented in the acceptance index.

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
application's 100-tool limit is unchanged. The current KVMem-on engine passed
all 36 pages with the exact saved file and without copying the reference into
prior text. The paired dense run timed out under the same deadline; it also
violated the intermediate-transcription rule. This workload is separate from
the three frozen short tasks.
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
ports. The latest paired trial completed all 36 pages with KVMem on; off reached
page 18 before the same 900s deadline. Run new trials with a fresh directory:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_long_suite.py \
  --kvmem on --guided-refresh both --run-dir apps/computer_use/.runtime/archive-suite-new
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

Edit/RichEdit input now uses an undoable `EM_REPLACESEL` operation on the focused
Unicode control, without taking clipboard ownership. It validates focus,
read-only/disabled state and NUL before dispatch; other windows retain paced
Unicode keyboard batches. Nine native-control checks passed for selection,
undo, Unicode, refusal cases and unchanged clipboard sequence. Actual Notepad,
Edge and forced-termination qualification is tracked in the acceptance index.
The previous clipboard-preserving implementation and its successful input and
18-task results remain baseline evidence; they do not qualify a revised
executor. See the Windows validation report for retained failures and hashes.

`windows_input_recovery.py` runs a dedicated VM fault test: it seeds five
clipboard formats, kills only the identified executor while a selected action
is durably dispatching, checks restart/duplicate refusal, explicitly reconciles
that known fixture action, and saves through the UI. It requires the current
executor to be deployed and all model/VM task suites to be stopped:

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/windows_input_recovery.py \
  --run-dir apps/computer_use/.runtime/input-death-new
```

The setup scripts in `vm/` are administrative test fixtures and are never
exposed to Computer MCP. Provisioning and the fault test use unique shared
script/request/result paths so rapid calls cannot accept an earlier SMB-cached
result. Production uncertain actions still require the user to inspect the
screen and explicitly reconcile them before continuing.

## Windows OSWorld campaign preparation

The ten-scenario plan and live preparation status are recorded in
`docs/computer_use_windows_execution_optimization_20261001.md` and
`docs/computer_use_windows_execution_progress_20261001.json`. Formal on/off
runs require qualified task initializers, graders, and a new frozen snapshot;
the earlier Notepad/browser suite is separate evidence.

The desktop schema now includes fixed launch targets for LibreOffice
Writer/Calc/Impress, Thunderbird, GIMP, and the pinned WPS Spreadsheets binary.
It exposes 73 fixed keyboard actions and bounded 100–2000 ms drag actions with
an optional Shift, Ctrl, or Alt modifier. A drag releases held input on error;
a durable marker also releases it after executor process death. The original
action stays uncertain and cannot be replayed. `windows_office_controls.py`
and `windows_drag_interruption.py` qualify these contracts in the real VM.

`vm/osworld-web.compose.yml` runs task-owned FormCraft, TeamChat, CloudCRM,
Calendar, and MailHub services from the pinned OSWorld-web source. Browser
access uses synthetic `.qw3.test` hostnames, a loopback/VM-private gateway, and
normal application APIs. Full-state/management endpoints are restricted to
the host's separate loopback ports. Set `QW3_WINDOWS_BRIDGE_IP` to the verified
VM bridge gateway; clear uppercase and lowercase HTTP/HTTPS/ALL_PROXY variables
when building or downloading. The shared frontend recipe disables npm audit
and funding network requests while retaining package-lock verification.

`osworld_windows/` contains host-only setup and grading code. It is deliberately
excluded from the native client bundle. Task 013 preparation hashes the pinned
PDF, initializes a fresh private form identity, and stages only public task
materials in Windows. Its grader preserves the upstream best-whole-solution
field score and separately requires actual submission and an unchanged form
for strict success. `qualify013` checks positive/negative outcomes against the
upstream comparator; `qualify013_gui` records a known-answer GUI submission.
These checks are not model results. Never give the agent evaluator fixtures,
expected answers, or host state-management interfaces.

`task_driver.py` records one native application task until completion, failure,
pause, budget exhaustion, or deadline. It does not grade the answer or resume
uncertain input. Ordinary terminal outcomes can advance a campaign only after
the desktop is stopped; infrastructure errors stop the queue. Full campaign
scheduling, scene restoration, and independent grading remain separate host
responsibilities.

`probe_compaction.py` exercises the real pinned Codex/bridge control plane with
scripted token usage. At the 950K threshold, the observed compaction request
includes the prior image and the following request contains the text summary
without that image. Both policies must use this same history behavior and
record compaction events. This probe is not a 950K-token GPU inference test.

## LaTeX software workflow

The current Windows candidate adds `setup_latex` and `repair_latex` with the
`latex-chinese-math` profile. Native contract checks passed; real model product
acceptance is tracked in `docs/computer_use_latex_product_20261003.md`.

Place a UTF-8 `main.tex` in `SoftwareRoot/projects/<project>` (or let a new setup
create the included Chinese/math example), then ask the local agent to configure
that project. The workflow installs a private portable MiKTeX 25.12 under
`SoftwareRoot/latex/<project>`, verifies the fixed assets in
`qw3_computer/latex_catalog.json`, and compiles in two new processes. It leaves
system PATH untouched and provides `open-latex-terminal.cmd` and
`compile-latex.cmd`. The result receipt and app show the produced PDF path.
Downloads ignore proxy environment variables and verify TLS and SHA-256.

Repair accepts only a matching managed project. It compiles the original source,
maps a missing file to the verified package archives, repairs at most two known
packages, and compiles again. Unknown failures retain their logs for takeover.
It cannot silently upgrade an arbitrary existing TeX distribution. Interrupted
installers remain uncertain and cannot be automatically restarted. Compilation
disables automatic package installation, shell escapes, and first-line option
parsing. This is a controlled installation workflow, not an OS sandbox for
untrusted TeX source; use a VM when opening documents you do not trust.

Run the product harness with `--scenario latex --thinking` and a qualified
client bundle. It restores the declared Windows baseline and records continuous
video; host Poppler tools independently inspect and render the actual PDFs.
Script qualification and native contracts are separate from model task passes.

The fixed Python setup/repair and LaTeX setup/missing-class repair scenarios each
passed three formal Windows VM trials, including independent artifacts, actual
launchers and application restart. These are controlled-software workflows;
arbitrary package sets, physical machines and ordinary-memory deployment remain
unqualified. The current 1M-context GPU profile used about 67 GiB in the LaTeX
trials. Full local recovery/capability/KVMem diagnostics and optional advisor
integration remain active work; the advisor is still disabled and unimplemented.
