# OSWorld-V2 local validation

Validated on 2026-09-03 and 2026-09-04 after the DeepSWE run completed.

## Prepared inputs

- OSWorld source, 108 gated task classes, 1061 gated assets, the QEMU VM,
  Docker runtime, and all three local GitLab images are present and pinned to
  the revisions and digests in `PROVENANCE.md`.
- `verify_setup.sh` passed in full before evaluation, including task and asset
  hashes, VM archive hash, Docker images, KVM access, and QW3 model files.
- The dependency audit loads all 108 tasks without error. Eight tasks require
  a residential proxy; the generated no-proxy task list contains the other
  100 tasks. Two tasks require the prepared local GitLab service.

## Evaluation checks

### Task 052: TravelHub booking flow

- Result directory:
  `/home/chaidi/qw3/benchmark/osworld_v2/results/smoke-20260903T142945Z`
- Completed all 15 configured model turns without an infrastructure error.
- This was the deliberately short `diagnostic-smoke-15` profile, not the
  formal OSWorld-V2 long-horizon QW3 profile.
- QW3 received screenshots, emitted native `computer_use` tool calls, and the
  adapter converted relative coordinates into correct 1920x1080 pyautogui
  actions.
- Final reward: `0.0`. The model repeatedly lost progress while handling a
  recurring limited-offer overlay and did not reach the requested room.

### Task 100: untimed jigsaw puzzle

- Result directory:
  `/home/chaidi/qw3/benchmark/osworld_v2/results/puzzle-smoke-20260903T144229Z`
- The VM, website, visual observations, multi-tool drag actions, and incremental
  evaluator all initialized successfully.
- The run was stopped after a deterministic zero-progress loop was established:
  the same piece alternated between two positions and the live reward remained
  `0.00`. Because this run was intentionally interrupted, it has no final
  `result.txt` and is not a completed benchmark sample.

## Screenshot-compaction latency experiment

- Experiment directory:
  `/home/chaidi/qw3/benchmark/osworld_v2/compaction_ab/20260903T_trace_v1`
- Same OSWorld-V2 task 052, one environment, eight turns per condition,
  temperature 0, 512 output tokens, and a 100-step text/action history.
- `compact`: `image_max=4`, `fold_size=2`; live-image sequence
  `1,2,3,4,3,4,3,4`, with folding events at steps 5 and 7.
- `no_compact`: `image_max=64`; live-image sequence
  `1,2,3,4,5,6,7,8`, with no folding event.
- Both runs used QW3's `plain` route with KVMem disabled. This result measures
  the adapter's screenshot compaction, not QW3 KVMem compaction/retrieval.

Before the first folding boundary (steps 1–4), the paired server-latency
difference averaged only 0.322 seconds in favor of compaction. From steps 5–8,
the compact condition averaged 41.873 seconds per model request versus 71.726
seconds without compaction: 29.853 seconds saved per step, or 41.6%, with a
mean paired speedup of 1.72x. Within the compact run, latency at the two event
boundaries changed by -10.886 seconds (-23.7%) at step 5 and -8.570 seconds
(-18.6%) at step 7 relative to the preceding step. These adjacent-step changes
are descriptive, because each step has different screen content.

Of the post-boundary paired saving, 27.830 seconds per step (93.2%) came from
server preprocessing and 2.023 seconds (6.8%) from engine generation. A simple
fit over the eight no-compaction samples associated each additional retained
screenshot with 11.313 seconds of server latency. The live trajectories and
screens can diverge between conditions. This slope and the paired deltas are
therefore engineering observations from one task and one sequential run, not a
multi-run confidence interval.

The audit verified, for each condition, eight structured agent traces, eight
step timing rows, eight action/trajectory rows, eight matched QW3 accounting
records, ten guest resource samples, the initial and all post-action images,
all eight exact model-input image hashes, all sanitized message records,
evaluation timing, result, runner/server logs, and the run manifest. Each
condition has a `trace_integrity_manifest.json` containing size and SHA-256 for
every captured file. The merged data, report, CSV, JSON, and latency plot are
under the experiment's `analysis/` directory.

Both conditions scored `0.0`; the run is valid for the controlled latency
comparison but is not evidence of task-solving quality.

## Go/no-go decision

The current QW3 build can technically execute OSWorld-V2 end to end, but the
current Qwen GUI-agent policy has not passed either smoke task. Do not start the
100-task or 108-task run with this configuration: first improve loop detection,
action planning, and handling of recurring overlays, then repeat the smoke.
The complete 108-task run also remains gated on valid credentials for the eight
official `proxy=true` tasks.

The prepared formal QW3 full/no-proxy runners now use the explicit
`osworld-v2-public-pyautogui-200` profile with 200 turns, following the official
public-evaluation pyautogui example. No 200-step task has yet been executed, so
the existing smoke and latency results must not be reported as formal-profile
scores.
