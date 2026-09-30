# Codex provider and Windows provisioning progress

Date: 2026-10-01, Asia/Shanghai. This is an intermediate record, not full
computer-use acceptance. The main goal remains active.

## Implementation and validation

Added `apps/computer_use`: a pinned Codex 0.159.2 stdio client, a serialized
Responses-to-QW3 streaming bridge, protocol tests, an actual Codex/MCP fixture
integration test, a GPU engine launcher and a real-model image/tool smoke.

The 11 Python checks passed outside the sandbox, including full-history image
preservation, tool-call identity, two actual Codex/MCP rounds, interruption of
a silent backend, backpressure, invalid requests, truncated streams, malformed
tool arguments and token-limit handling. The latest run also checked the
restricted child-process environment and 1M model catalog configuration.

The CUDA/FlashInfer build completed successfully with:

```sh
cmake --build build-dflash-integration -j 4
PYTHONPATH=apps/computer_use QW3_TEST_CODEX=1 .venv/bin/python \
  -m unittest discover -s apps/computer_use/tests -v
```

CPU baseline validation remains the previously recorded 44 CTests and two
benchmark-helper Python checks. No claim is made that every newly built GPU
test executable has run.

## Real QW3 + DFlash image/tool smoke

Successful engine run: `apps/computer_use/.runtime/dflash-smoke-04`.
Successful Codex run: `apps/computer_use/.runtime/codex-gpu-smoke-02`.
These ignored local directories retain the exact manifest, server log,
Codex events and structured result. The binary hash was:

`e14b429817f42cedee5659c786087f0265f6335a58073d884a2108782ca9a2ad`

Configuration: Qwen3.8-27B Q8_0 target, matching BF16 DFlash draft, CUDA vision,
FP8 KV, serialized serving, K128K/G64K, 1M logical context, 32GiB CPU tier,
128-token blocks/four 32-token slices, sub-block-mean-k message expansion and
multimodal prefix reuse. Native MTP weights were explicitly skipped. This
initial protocol profile uses `enable_thinking=false`.

The model identified the red input image, called the fixture screenshot tool
twice in sequence, and correctly reported observation numbers 1/2 and blue
screenshots. Both MCP calls completed. Total smoke time was 3.192 seconds,
including the Codex startup/turn flow; it is one small synthetic image case,
not a desktop action-latency benchmark.

| Request | Server TTFT, ms | Server total, ms | DFlash drafted / accepted | Fallback |
| --- | ---: | ---: | ---: | --- |
| User image → first tool | 1066.82 | 1505.25 | 68 / 25 | false |
| First tool image → second tool | 233.91 | 504.76 | 47 / 26 | false |
| Second tool image → final answer | 236.23 | 652.76 | 74 / 38 | false |

The latter two requests reported `kvmem_reuse=1664`. The final request had
three image cache hits and zero misses. History was only 2140 prompt tokens:
this validates reuse, not actual sparse retrieval past the K128K threshold.
The current server logged `visual_history_policy=pin-all`; scalable image
history retention/retrieval remains an explicit later test item.

A post-smoke whole-device snapshot reported 43,308 MiB used out of 97,887 MiB.
This is resident device usage at one instant, not a sampled peak, allocator
bound or proof of fit on a 48GB GPU. There is no 48GB acceptance cap.

All initial failures were retained. Engine attempts 01–03 failed configuration
validation for missing sub-block scoring, incompatible scoring slice size and
missing CPU raw-K capacity respectively. The launcher now encodes those
dependencies. Codex smoke 01 failed because the engine had exited. Smoke 02
is the first successful actual-model attempt; failures are not counted as
successful runs.

## Windows VM

Created only `qw3-computer-windows`, owned by label
`com.qw3.computer-use.owner=codex-goal-20260930`, using:

`dockurr/windows@sha256:0cff9eb0e7aee9953e55bc682852ca4fdca233145a58ae1ec94f0b0c01a2ed30`

Windows 11 Enterprise evaluation media downloaded from Microsoft's servers.
The actual aria2 process had no proxy environment or proxy arguments and used
`--no-conf=true`. Future launches explicitly clear both cases of HTTP/HTTPS/ALL
proxy variables. Host/global proxy settings were not changed.

The VM uses a new sparse 64GiB virtual disk, dedicated credentials, four vCPUs,
and loopback-only published ports. Its first boot auto-reduced guest RAM from
the requested 8GiB to 3683MiB because downloaded ISO cache counted against the
initial 12GiB container cap. The cap was raised to 20GiB; after installation,
restart and verify the actual guest allocation before running task measurements.
Do not label the first boot as an 8GiB run.

VNC screenshots confirmed the real Windows installer, first at 84% and then,
after reboot, at its next 33% installation stage. The initial 84% is not an
overall completion claim. The VM has not yet passed desktop-task acceptance.

## Remaining work

- Build the bounded desktop executor, Computer MCP gateway and task UI.
- Verify stale observations, action IDs/uncertain delivery, pause/takeover,
  application restart and no-progress recovery against the Windows VM.
- Freeze a clean installed snapshot and verify independent file/state results
  for all three initial tasks, three repetitions each.
- Exercise real retrieval, growing screenshots, QW3 cancellation/resume and
  decoder comparisons; measure memory peak and task latency distributions.
- Complete Windows client, physical-desktop and macOS acceptance when the
  applicable targets are available. Mac connectivity remains pending and does
  not block the Windows VM work.

See `apps/computer_use/README.md` for reproduction commands and the supported
protocol subset. A mock tool or image-color smoke is not a Windows task test.
