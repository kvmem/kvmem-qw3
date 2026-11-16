# Updated priority: add InfLLM to the existing long-running agent setting

User steering on 2026-09-12 supersedes the three-method BF16 pilot and its queued
formal runs. Reuse completed KVMem and ordinary qw3 trajectories. Add InfLLM;
do not automatically rerun KVMem/dense or expand FlexiCache integration.

The research question is whether existing retrieval plus offloading is sufficient
in the KVMem workload: a long-running agent repeatedly observes tool results,
diagnoses, acts, and revisits its growing history. Both KV methods can remove
client summaries. Their comparison therefore needs task capability, ordinary
step overhead, maintenance stalls, decoding and total task time, not just the
absence of compaction.

## Experiment and implementation boundaries

- Use original DeepSWE tasks, tools, repository states and hidden judges, with
  the same task/seed cohort as the reused results. Do not inject artificial
  history, force a smaller context to trigger maintenance, or substitute the
  InfLLM paper's long-document benchmarks for the main agent experiment.
- First retain the previously selected first four tasks, subject to confirming
  completed comparable historical arms. Choose a common historical seed before
  examining outcomes; seed 1000 is the first candidate, not the pilot's seed 73.
- Freeze exact source paths, model/precision, client version, sampling, limits,
  hardware, and metric definitions from raw launch metadata. Preserve all old
  results and explicitly record unresolved metadata rather than inferring it.
- InfLLM should preserve its author's selection, representative tokens, attention,
  position handling, cache, offload and asynchronous operations. Module oracle
  tests validate this adaptation; they are not the main evaluation workload.
- Match InfLLM to the existing system configuration where technically supported.
  Existing inspected DeepSWE logs use Q8_0 weights, FP8 KV, MTP chain 4, KVMem
  selection 131072, and a 3M logical history. Current InfLLM uses BF16 weights,
  FP16 KV, no speculative decoding, approximately 32K attention, and 512K logical
  history. These differences are unresolved and preclude a matched speedup claim.
- Do not pretend InfLLM's always-selected context is identical to KVMem's
  selection plus growing generation window. Record physical memory, effective
  active KV and algorithm-specific budgets separately. Exact same block size
  is not required across distinct algorithms.
- Avoid changing the published setting merely to fit currently spare GPU memory.
  Shared GPU is permitted for integration and task behavior; formal timing must
  have an appropriate hardware/load match to the reused measurements.
- The current InfLLM integration handles text. OSWorld additionally needs verified
  multimodal input and position handling; do not present text-only runs as an
  OSWorld comparison. Start with the existing DeepSWE text cohort.

## Transition and next work

The current full DeepSWE InfLLM arm is preserved as integration evidence under
its actual BF16/32K/seed-73 configuration. It is not a matched new baseline for
the old Q8/FP8 runs. The old three-method scheduler is stopped without stopping
that arm; an identity-checked watcher retires it when the arm exits. The waiting
two-method formal measurement queue was cancelled before taking measurements.
Frozen old files and evidence remain intact.

Next: complete a historical configuration/provenance manifest; identify the
minimum faithful InfLLM adapter work needed for Q8_0, FP8 and session/budget
compatibility; validate it on an existing text trajectory; then run only InfLLM
on the selected historical tasks. Do not claim configuration alignment or launch
a replacement matched campaign until those compatibility checks actually pass.
Any unavoidable precision/backend/acceleration differences must be disclosed;
no new KVMem or dense run is authorized by this revised scheduling plan.
