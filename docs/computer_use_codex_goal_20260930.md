# Codex + QW3 computer use build and validation goal

Created: 2026-09-30, Asia/Shanghai. Status: active.

## Objective and agreed scope

Build a computer use application around Codex app-server, QW3/KVMem and
DFlash speculative decoding. First validate desktop operation in repeatable
virtual machines, with inference on the current NVIDIA GPU host. Use its
available 96GB-class GPU capacity and report measured memory use; 48GB is not
a hard constraint. Follow with physical-desktop acceptance; deployment on
ordinary Windows/Mac machines remains the subsequent product target.

Latest scope update, 2026-09-30: the user explicitly relaxed the original 48G
limit. This supersedes the memory constraint in the initially created goal.
Prioritize correct operation and measured performance. Treat lower-memory
profiles as separate later evaluations, not a gate for the first application.

The user selected Codex as the runtime harness. Do not substitute a new custom
agent loop or migrate to llama.cpp as the primary implementation for this goal.
Use Qwen3.8-27B Q8_0 target weights, FP8 target KV and the matching DFlash draft.
DFlash and native MTP are alternative draft models; run native MTP and target-only
decoding as separate controls, never simultaneously with DFlash.

Latest platform priority, 2026-09-30: validate Windows in a VM first, as
requested by the user. Mac connectivity can be prepared later and does not
block the Codex/provider/Windows work.

Download preference: clear lowercase/uppercase HTTP, HTTPS and ALL proxy
variables for VM image downloads. Do not change the host's global proxy settings
or unrelated containers. The initial aria2 Windows ISO download had no proxy
environment variables; the VM launcher also explicitly clears them for later runs.

Comparison scope update, 2026-10-01: the user also requires KVMem enabled versus
disabled comparison. Use a KVMem on/off × none/native-MTP/DFlash matrix with
the same frozen task, image and history inputs. Compare success, latency,
throughput and CPU/GPU resources. Include cold fixed-input replay to separate
retrieval effects from prefix-reuse effects, alongside the actual application
flows. Report long-history capacity failures as such; do not truncate only one
side or compare different inputs as equivalent runs.

## Existing baseline and evidence

- Preserve the existing source changes in a local Git checkpoint before new
  implementation. Include required new source, tests, scripts and vendored
  utf8proc files. Leave browser profiles, private machine configuration, model
  weights and bulk experiment artifacts out of the checkpoint.
- The current source already contains the 2026-09-29 DFlash integration. Use
  `docs/dflash2_integration.md` for its provenance and limitations. Earlier
  documents describing this checkout as lacking the integration are stale.
- `docs/dflash2_single_request_results_20260927.md` describes text-only replay
  on a 96GB physical GPU. The sampled peak near 48,000,000,000 bytes is not a
  hard allocation bound or proof of multimodal capacity on a 48G card.
- Reuse `scripts/osworld_v2/` and the pinned OSWorld-V2 environment where useful.
  Existing benchmark outcomes are historical evidence, not acceptance of the
  new Codex application.
- Initial sandbox inspection could not access the NVIDIA driver or `/dev/kvm`.
  The user confirmed the current host as the test machine. An approved
  read-only check outside the sandbox found an NVIDIA RTX PRO 6000 Blackwell
  Server Edition (97,887 MiB reported memory, 14 MiB used at inspection), no
  reported compute processes, and `/dev/kvm`. GPU/KVM access is therefore
  available through the host execution path. Docker has existing experiment
  containers; their presence is not authorization to stop or replace them.
  The user subsequently allowed using the larger available GPU capacity.
  A macOS test machine has not yet been identified.

## Architecture

The application owns its interface, task record and lifecycle. It starts a
pinned Codex app-server over stdio. Codex talks to a local Responses-compatible
adapter backed by QW3 Chat Completions, and to a Computer MCP server. The MCP
server operates a desktop executor inside the target VM. Inference remains on
the GPU host, avoiding GPU passthrough as an initial dependency. Replace the VM
executor connection with a physical desktop connection during later acceptance.

All model inference, screenshots, task history and memory stay on the user's
machines. No cloud-model fallback, remote judge or external telemetry is part
of this goal. Tasks may access explicitly selected websites; inference locality
does not imply that every target website must be offline. Use local fixture
applications for reproducible offline tests.

Keep application event history and screenshot files as the replay authority.
KVMem is derived execution state, scoped to the model, configuration and task.
History restoration must never blindly repeat a potentially completed action.

## Milestones and acceptance

### M0: checkpoint and build baseline

1. Commit the existing source changes with their necessary new dependencies.
2. Build a fresh CPU Release configuration and run registered host CTests.
3. Run the changed continuous-batching benchmark helper's Python tests and
   syntax-check the selected Python sources without executing experiment jobs.
4. Record the commit, commands, results and unavailable GPU checks.

### M1: Codex provider bridge

Implement the Responses subset used by the pinned Codex version. Preserve text,
image inputs, function schemas, call IDs, tool outputs, stable history and
streaming completion/error events. Unsupported request semantics must produce
an explicit error. Do not advertise complete Responses API compatibility.

Acceptance covers text and image requests, image tool results, multiple tool
rounds, cancellation, malformed inputs, backend failure and stream termination.
Use contract fixtures and a controlled local backend first, then demonstrate
the same cycle through real Codex and QW3. Stub success is not model acceptance.
Verify that model capabilities allow images and that Codex history management
does not discard the evidence required by the selected KVMem strategy.

### M2: Computer MCP and VM application

Expose observation and bounded desktop actions with typed arguments. Associate
actions with their observation and desktop identity. Validate stale observations,
coordinates/targets and application scope before dispatch. Record the action
and its post-action observation. Provide pause, cancellation and human takeover.
Detect repeated actions without progress and return a recoverable result.

The application must display task progress, the target desktop, current action,
pause/stop controls and final result evidence. The VM executor and physical
executor use the same application-level contract. Reuse existing OSWorld
infrastructure without modifying archived benchmark evidence.

Acceptance covers observe/click/type/key/scroll, screenshots, visible state
changes, unavailable desktops, tool errors, cancellation during an action,
takeover/resume and duplicate-call handling. A desktop operation without a
natural idempotency guarantee requires observation-based reconciliation after
uncertain delivery, not an unconditional retry.

### M3: multimodal KVMem + DFlash correctness and resource measurement

Start with one active desktop task and serialized multimodal serving. The
existing K128K/G64K text-replay configuration is a starting candidate for the
96GB-class host, not a measured multimodal capacity claim. Record the exact
device capacity and application usage in bytes, with consistent GB/GiB units.
There is no 48GB pass/fail threshold. Smaller working sets and generation
reserves can be evaluated separately after the functional baseline is stable.

Account for target/draft weights, vision tower, image embeddings, active KV,
indexes, verifier buffers and transient allocations. Measure first-image
encoding, repeated image reuse, new images, growing history, real retrieval,
tool continuation, cancellation and resume. Compare target-only, native MTP and
DFlash separately under the same task/input policy. Capture actual speculation
acceptance and fallback counters; a configured flag alone is insufficient.

Acceptance requires correct tool/image round trips, valid outputs and state
restoration, no OOM, and measured memory fitting the available GPU with
practical headroom for the tested workload.
Sampled whole-device usage and allocator measurements have different scopes;
report both where available and do not present sampling as a hard guarantee.
Report screenshot-to-action latency, TTFT, action latency P50/P95, vision time,
decode time, retrieval/replay time, CPU RAM, GPU peak and task outcomes.
The KVMem on/off decoder matrix above is now required acceptance evidence.

### M4: repeatable VM and physical-desktop validation

Freeze task inputs, application versions, VM snapshot identity and acceptance
checks before measuring. Initial deterministic tasks should cover a local
browser form saved to a file, a local document editing/saving flow, and a
multi-application copy/verification flow. Verify delivered files or application
state independently of the model's claim of completion.

Run at least three repetitions of each initial task, retaining every failure.
Add a long-history task that actually crosses the configured KVMem retrieval
threshold, plus pause/resume, application restart and no-progress recovery cases.
Separate fixed-input engine replay from live task success; neither substitutes
for the other. The deterministic initial suite must pass in every repetition
before broader exploratory OSWorld testing.

Validate the Windows executor and client in a Windows VM, and the macOS executor
and client on an Apple-hosted macOS VM or physical Mac. Linux VM results cannot
be described as Windows/macOS validation. Run the same core task checks on the
accessible physical target desktops, including scaling, input methods, focus,
permissions and takeover. Missing target hardware remains an open acceptance
item, not a passed or silently removed test.

## Implementation order and completion rule

Work in order M0 -> M1 -> M2 -> M3 -> M4. Protocol, application and executor
development can proceed while GPU/VM host information is pending. Freeze the
source/model/binary identities used by each real run; do not replace running
experiments or reuse occupied GPUs without checking their ownership.

The goal is complete only with a reviewable application, reproducible launch
instructions, test evidence and explicitly closed required acceptance items.
Compilation, mocked protocol tests or old benchmark reports alone do not
complete it. Record unavailable infrastructure and continue independent work;
do not claim GPU, VM or physical-machine tests that have not run.

## Progress at baseline checkpoint

- M0 source checkpoint: the Git commit containing this goal document preserves
  the existing runtime work and its required source dependencies.
- Fresh CPU Release build passed in `/tmp/qw3-checkpoint-build-20260930`, using
  `QW3_ENABLE_CUDA=OFF`, `QW3_ENABLE_FLASHINFER=OFF`, `QW3_BUILD_TESTS=ON` and
  `cmake --build ... -j 4`.
- All 44 registered CTest checks passed: 43 inside the sandbox, followed by
  the local HTTP disconnect check outside it. The initial disconnect failure
  was at loopback port binding; the identical executable passed on the host.
- Both `tests/test_continuous_batching_benchmark.py` unit tests passed.
- Python AST parsing passed for the 45 Python files included in the checkpoint.
- The project-source whitespace check passed. The vendored
  `third_party/utf8proc/utf8proc_data.c` retains upstream trailing whitespace
  and its original content identity; it is excluded from that formatting check.
  GPU inference and VM tasks have not run for this new goal. M1 provider
  implementation is the next milestone.

## References

Progress after M0: the provider prototype and a real Codex → QW3/DFlash →
two-round image MCP smoke are working. Windows VM installation is complete.
See `computer_use_codex_m1_20261001.md` for exact successes, retained failures,
resource observations and remaining acceptance. The goal is still active.

Progress 2026-10-01: the local UI, durable Computer MCP boundary and Windows
executor now work with real Codex and QW3/DFlash. Three Windows tasks produced
independently verified files; cross-application copy/paste also passed a real
pause/resume cycle. These are single smoke trials, not the three-repetition
snapshot acceptance. See `computer_use_codex_m2_20261001.md`. The short-history
KVMem on/off × decoder matrix completed 54/54 frozen-input requests; see
`computer_use_kvmem_comparison_20261001.md` for actual timings and limits.
Native MTP at 1M context exceeded the initial 32GiB CPU backing configuration;
the common comparison configuration now reserves 48GiB CPU tier capacity.
This does not reinstate the retired 48GB GPU limit. Long-history retrieval,
native Windows/Mac client packaging and physical-desktop acceptance remain open.

Further progress 2026-10-01: above-budget historical-image diagnostics passed
at 158,911 tokens, including a changed-image negative control that invalidated
prefix reuse. A matched DFlash KVMem on/off replay passed all four positive and
negative checks. Real Codex preserved eight image observations and all archive
notes through 168,845 tokens and correctly answered the old-image question with
actual sparse selection. See `computer_use_long_history_20261001.md`; synthetic
archive diagnostics do not substitute for live desktop task acceptance. The
application now publishes task completion only after collecting output evidence.

Windows repetition update: `windows-suite-01` stopped at its first failed trial
after seven successes. The browser failure exposed inaccurate model click
coordinates, so the new observer adds generic Windows accessibility bounds
alongside screenshots. The no-progress check now uses foreground-window pixels
after a real test exposed taskbar animation masking an unchanged application.
Real executor lost-reply recovery passed across an executor restart without
replaying the action. See `computer_use_windows_validation_20261001.md` for
failure evidence and remaining qualification; the full live on/off suite has
not passed and the goal remains active.

The corrected browser pilot subsequently passed with 10 actions and verified
JSON. The offline Windows x64 client also passed all 29 contract/lifecycle tests
with native CPython 3.13.13 and Codex 0.159.2, exit code 0, including image MCP
round trips and interruption against a scripted backend. Native-client desktop
execution with the actual GPU model is still a separate pending check.

Real application crash/resume also passed on the Linux host against the Windows
VM: restart remained paused until explicit resume, then the original task
completed with the exact saved copy code. The corrected 18-trial live on/off
suite has started from the new snapshot; full repeated acceptance is still open.

The second suite subsequently finished with 17/18 passed: on 9/9, off 8/9.
The last cross-app task timed out after inefficient wheel navigation and a
Save As foreground transition. All attempts and per-trial resource/action
metrics are retained. The current repair makes wheel direction and target
coordinates explicit, returns a fresh observation after a confirmed rejected
action, and waits for scroll animation. The real executor probe and 31 local
contracts pass; a new model-driven suite is still required. The queued native
pilot correctly did not start after the failed prerequisite. See the updated
Windows validation report; this does not close repeated acceptance.

The scroll-compatible offline client (bundle 04) subsequently passed all 31
tests on actual Windows with exit code 0. A new clean, powered-off snapshot
`qw3-scroll-20261001` includes that client and the revised executor. Native
GPU-backed task execution and the repeated model suite remain pending.

Latest qualification update: the 158,911-token frozen history now covers all
six decoder/KVMem profiles, with **12/12** positive/negative checks passed
(one repetition per case). All profiles use the same binary, launcher and
frozen input hashes. KVMem lowered sampled GPU peak but increased cold request
time in this diagnostic; see `computer_use_long_matrix_20261001.csv`.

The repaired off-mode cross-app pilot passed in 135.96 seconds. Native Windows
client execution with the real GPU then passed the browser pilot in 40.66
seconds, including exact file verification and a visually inspected native UI.
Native app crash/resume also passed in 63.12 seconds, preserving task/thread
identity and staying paused until explicit resume. These close the corresponding
single-run native cases. The new native 18-trial on/off matrix is running from
`qw3-scroll-20261001`; it has not yet passed. A serial queue will audit that
matrix before starting the real 36-page archive task with KVMem on and off.
The goal remains active; repeated reliability, live above-budget execution,
further allocator accounting, Mac and physical targets remain open.

- Codex app-server: https://learn.chatgpt.com/docs/app-server
- Codex provider configuration:
  https://learn.chatgpt.com/docs/config-file/config-reference
- Codex MCP: https://learn.chatgpt.com/docs/extend/mcp
- Local integration: `docs/dflash2_integration.md`
- Existing VM harness: `scripts/osworld_v2/README.md`
- DFlash memory evidence: `docs/dflash2_single_request_results_20260927.md`


Latest update: CUDA allocation diagnostics and literal-input repair (2026-10-01).
The explicit memory-tracing launcher is now committed (`e924840`), and its paired
DFlash long-input run passed all four visual checks plus both trace-consistency
checks. Event-sampled device peaks were 43.42 GiB on and 67.14 GiB off. The scopes
and untraced allocation categories remain explicit in the long-history report;
this is not a complete allocator census or a 48GB physical-device qualification.

The native 18-trial run stopped after nine passes and an off-mode Notepad
failure caused by real Unicode input corruption. Direct saved-file probes
reproduced it without a model. The repaired executor (`53cd0bf`) passed its
final five file/browser checks and preservation of five clipboard formats.
It is frozen in `qw3-type-20261001`, with native client bundle 04 unchanged.
The previously failing native off-mode Notepad task now passes in 111.84 seconds,
with 11 completed actions, an exact 46-byte file, 1,803/3,341 DFlash drafted
tokens accepted and no fallback. Its single-trial evidence check passed; the
full matrix gate intentionally remains false for a one-case pilot.

The success gate has started `windows-native-suite-02`: 18 native Windows
on/off trials, followed only after successful matrix audit by the paired live
archive runs. Application sources and snapshots remain frozen for this queue.
The UI screenshot and complete pilot evidence are retained under
`windows-native-type-pilot-01`. Required repeated acceptance, live long desktop
history, forced executor termination during temporary clipboard ownership,
macOS and physical-desktop acceptance remain open. The goal remains active.

Evidence index and target preparation (2026-10-01):
`computer_use_acceptance_20261001.md` and its JSON companion map completed
requirements to retained artifacts and keep the outstanding gates explicit.
A consolidated CPU recheck passed all 44 CTests and both benchmark-helper tests;
prior CTest logs were retained. The read-only macOS preflight helper and
`computer_use_mac_test_access_20261001.md` describe target access without asking
for passwords or private keys. Only syntax/help/non-Mac refusal have been
verified; no Apple API or Mac desktop test is claimed.

Documentation and this independent Mac helper may be committed while the
Windows queue is running: none of the frozen `apps/computer_use` sources,
engine/model assets, guest state or snapshot is changed. Recorded Git commits
can therefore differ between trial manifests; their application-source and
binary hashes remain the qualification identities. The native suite remains
in progress, with no final acceptance declaration.
