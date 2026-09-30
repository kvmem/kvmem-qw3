# Local Codex computer use

Work in progress: the local UI, Codex provider bridge, Computer MCP and Windows
VM executor work end to end. Three real-model Windows smoke tasks have produced
independently verified files, including one pause/resume run. Repeated trials
from a clean snapshot, long-history retrieval and physical-machine acceptance
remain open. See `../../docs/computer_use_codex_m2_20261001.md` for evidence.
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
Image-budget compaction is disabled. Long-history retrieval and Codex's actual
behavior near the context limit still need measurement; short-history prefix
reuse does not qualify them.

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

The Windows guest currently runs the executor; the UI, Codex and GPU model run
on Linux. A packaged native Windows/Mac application and physical-desktop tests
are separate qualification steps. The dedicated VM test account is elevated;
this prototype is not qualified as a general-purpose desktop permission sandbox.

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

Sources: [Codex app-server](https://learn.chatgpt.com/docs/app-server),
[Codex config](https://learn.chatgpt.com/docs/config-file/config-reference),
[Dockur Windows](https://github.com/dockur/windows),
[Microsoft evaluation media](https://www.microsoft.com/en-us/evalcenter/download-windows-11-enterprise).
