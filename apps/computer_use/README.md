# Local Codex computer use

Work in progress: the provider bridge and real-model image/tool smoke are
working. Windows VM installation is provisioned. The desktop executor, task UI,
repeatable Windows task suite and physical-machine acceptance remain open.
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

The initial profile uses FP8 KV, K128K/G64K, 1M logical context, a 32GiB CPU
tier, 128-token blocks with four 32-token scoring slices, message-level
retrieval and multimodal prefix reuse. It uses the available 96GB-class GPU;
48GB is not an acceptance limit. `--method none` and `--method mtp` select
separate controls; neither control has been measured for this app yet.

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
VNC `15900`, and the planned desktop executor `18082`. The VM has no GPU
passthrough. The GPU model remains on the Linux host.

The bootstrap runs in the VM's test account desktop. It waits for the reviewed
`executor.ps1` and token file in the dedicated shared directory; the executor
is not implemented yet. Windows/Mac endpoint testing and inference running
entirely on an ordinary Windows/Mac machine are separate qualification steps.

Sources: [Codex app-server](https://learn.chatgpt.com/docs/app-server),
[Codex config](https://learn.chatgpt.com/docs/config-file/config-reference),
[Dockur Windows](https://github.com/dockur/windows),
[Microsoft evaluation media](https://www.microsoft.com/en-us/evalcenter/download-windows-11-enterprise).
