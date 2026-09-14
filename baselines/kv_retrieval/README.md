# KV retrieval baselines for long-running KVMem agents

## Current experiment: InfiniGen throughput

On 2026-09-12 the user canceled further InfLLM testing, keeping the observed
throughput, and switched to InfiniGen. The InfLLM controller, model server and
owned agent container have stopped. SuperJSON is user-canceled, Koota completed
with a zero full-pass score, Awilix retains its reviewed protocol failure, and
Textual was not started. These are not four completed accuracy trials.
Preserved metrics: [JSON](results/infllm_small_deepswe4_s1000_20260912/infllm_preserved_results_20260912.json)
and [CSV](results/infllm_small_deepswe4_s1000_20260912/infllm_preserved_throughput_20260912.csv).

InfiniGen support remains inside the independent `infinigen/InfiniGen` fork.
The completed throughput run used real 8K/32K/64K/120K recorded prefixes, Q8_0
weights, FP8 KV, batch one, no MTP, and identical recorded continuations for
InfiniGen and the same-backend dense control. Two repetitions collected raw
forward timings and sampled GPU ownership. This is not an online task accuracy
test. See [final report](infinigen/InfiniGen/qwen/results/exclusive_20260912_v5/REPORT.md), [summary CSV](infinigen/InfiniGen/qwen/results/exclusive_20260912_v5/summary.csv), and [run state](infinigen/InfiniGen/qwen/results/exclusive_20260912_v5/state.json).

## Preserved InfLLM experiment

The preceding experiment added **author-small InfLLM** to four existing DeepSWE
agent tasks at **seed 1000**. It reuses completed KVMem and ordinary qw3 runs;
it does not rerun those methods or replace live agent work with document QA.
The user canceled the large-window trial because of its low observed throughput
and requested small-window tests only. The interrupted large trial is preserved
as a diagnostic, not a scored task outcome. FlexiCache and the older BF16/FP16,
seed-73 integration pilot are also outside this campaign.

### Frozen settings

| Setting | InfLLM author-small window |
|---|---:|
| Initial/sink tokens | 128 |
| Local tokens | 4096 |
| Retrieved blocks × block size | 16 × 128 |
| Maximum active tokens | **6272** |
| GPU cached blocks | 32 |
| Logical history capacity | 3145728 |
| Weights | Existing Q8_0 GGUF |
| Persistent KV / assembled attention KV | FP8 / FP16 |
| Thinking guard / maximum output | 28672 / 32000 |
| Temperature / top-p / top-k / seed | 1 / .95 / 20 / 1000 |
| Client | Claude Code 2.1.148, original archived agent script |
| Client compaction / subagents | Off / off |
| Historical KV backing | CPU, no NVMe |

The active window is intentionally smaller than the historical KVMem capacity
ceiling (131072 selection + 65536 generation). This campaign tests the author's
small-window tradeoff in the existing agent setting. It does not establish
large-window InfLLM accuracy or an equal-capacity algorithm speedup.

The four original tasks are:

1. superjson-error-stack-serialization
2. textual-richlog-follow-state
3. koota-composite-trait-aspects
4. awilix-async-container-initialization

The small variant runs complete agent trajectories with original tools and hidden
verifiers, without injected history or padding. Preserve unsuccessful attempts and distinguish infrastructure failures
from valid zero-reward task outcomes. Fractional hidden-test credit is not a
full task pass.

Full protocol: [SMALL_WINDOW_AGENT_COMPARISON.md](reports/SMALL_WINDOW_AGENT_COMPARISON.md).
Frozen plan: [campaign configuration](configs/infllm_small_deepswe4_s1000_20260912.json).

To use the authorized spare GPU memory, later arms can run on an independent
port alongside the serial controller (at most two model servers). Awilix ended
with a tool-call protocol error and its original verifier ran; its
[terminal review](results/infllm_small_deepswe4_s1000_20260912/terminal_reviews/awilix-async-container-initialization.json)
and raw outcome are retained. Koota completed in that reserved slot. These arms
use the same frozen arm harness, seed and original output paths.
The serial controller skips completed arms, and its incomplete-attempt guard
must be respected if it reaches a still-running reserved arm. Scheduling and
CPU observer changes leave all 45 frozen model/task dependencies unchanged.

## Implementation and validation

- `upstream/`: pinned, pristine author repositories; original InfLLM retrieval,
  representatives, positional rules, LRU and offloading remain in use.
- `kv_baselines/`: model boundary, hybrid recurrent cache, session recovery,
  sampling and protocol bridge. Only full-attention layers use InfLLM.
- `compat_generated/`: inspectable InfLLM head-dimension/partial-RoPE compatibility
  changes, without edits to the author's source checkout.
- `alignment/`: current lossless GGUF Q8 layout adapter and native Q8 projection
  bridge, FP8 persistent KV, branch recovery and archived original agent script.
- `serve_aligned_infllm.py`, `run_aligned_infllm_arm.py`,
  `run_window_campaign.py`: frozen serving and four-trial execution path.
- `compare_agent_windows.py`: joins new trials with eight archived native trials
  and exports per-task/per-request data. Three-method paired statistics require
  complete, valid outcomes from every method on the same task.
- `observe_window_campaign.py`, `observe_window_gpu_load.py`: separate CPU-side
  progress/report and GPU telemetry observers; they never restart model jobs.
- `reports/`: validation/provenance, including preserved unsuccessful attempts.

The current GGUF SHA256 matches the historical locked weight hash. Q8 mapping
checks, same-quantized-input FP8 retrieval/output checks, real-model short
inference, full-model tail-rewrite recovery and a 202752-token component fixture
passed. These checks establish the exercised adaptation properties, not task
accuracy, native-model bitwise parity or a long-sequence performance result.

Validation evidence:

- [Current weight identity](reports/aligned_current_gguf_identity_20260912.json)
- [Q8/FP8 model and recovery validation](reports/aligned_q8_fp8_validation_20260912_v2.json)
- [Long-window component validation](reports/fp8_long_window_components_20260912.json)

## Reading current results

Campaign directory:
`results/infllm_small_deepswe4_s1000_20260912/`

- `state.json`: controller state and completed trial paths.
- `progress.json`: independently verified live-process and stream progress.
- `<task>/<variant>/attempt_01/result.json`: trial outcome and provenance.
- `comparison/trials.csv`: task scores, time, tokens and memory; incomplete
  outcomes stay missing rather than becoming failures or zero-duration tasks.
- `comparison/server_steps.csv`: recorded per-request timing, including cold
  starts. No latency trimming is applied.
- `comparison/comparison.json`, `comparison/README.md`: joined results and limits.
- `gpu_process_utilization.jsonl`: per-process GPU activity samples, starting
  near launch of this small-window campaign; missing activity remains unknown.
- `parallel_arms/*.json`: independently verified reserved-arm progress.
- `gpu_overlap/requests.csv`: sampled co-residency/activity per raw API request;
  regenerate with `audit_window_gpu_overlap.py --campaign <campaign>`. These
  observations neither adjust latency nor exclude primary trial outcomes.

Raw server traces and client events remain under each attempt. Live streaming
character/event counts are not model token counts. Request-level generation
metrics appear when a reply completes; an unchanged completed-request count
alone does not establish a stalled model.

The active model/controller and their frozen dependencies must remain fixed
within this campaign. Auxiliary report/observer changes are recorded separately.
Inspect incomplete attempts before retrying; do not overwrite them or start a
second campaign in the same output directory.

## Interpretation and resource policy

Call this method **InfLLM adapted to Qwen3.8**, not an unmodified published model
result. It uses HF/FLA BF16 intermediates and no MTP; historical native qw3/KVMem
used an FP32 residual stream and MTP chain4. Equal numeric seeds do not imply
identical RNG streams across backends. The small variant uses the author retrieval policy, but adapts the model and KV
storage boundary; generated trajectories can differ from the historical runs.

Current runs use spare GPU memory alongside another workload. Historical GPU
load is not fully verified either; the ordinary qw3 batch explicitly permitted
nonexclusive use. Report the observed times, but do not present their ratios as
pure algorithm speedups. Four tasks and one seed provide pilot evidence.

The separate CPU block-bookkeeping diagnostic exercises the author's exact
control flow with inert copies. It is not measured agent retrieval latency and
must not be multiplied by the layer count and reported as actual inference time.
See [its measurement limits](reports/AUTHOR_BLOCK_BOOKKEEPING_DIAGNOSTIC.md).

Use the isolated `baselines/kv_retrieval/.venv/bin/python`. Do not modify the
shared environment or stop unrelated workloads. Downloads unset all uppercase
and lowercase HTTP/HTTPS/ALL proxy variables; HF mirror is permitted. Earlier
integration scripts and reports are retained as provenance, not current run
instructions.
