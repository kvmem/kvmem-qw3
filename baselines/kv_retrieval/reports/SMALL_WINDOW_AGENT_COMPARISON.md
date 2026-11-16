# Current experiment: author-small InfLLM only

The user's latest instruction on 2026-09-12 stops large-active-window testing
because its observed throughput is too low. This supersedes the earlier
two-window objective. Complete **four small-window trials**, seed1000, and
compare with the existing four KVMem plus four ordinary qw3 trials. Do not
restart the large window or add fresh native trials.

Use 4096 local tokens, 128 sink tokens, 16 retrieved blocks of 128 tokens:
**6272 total active tokens**, with 32 GPU-cached blocks and a 3145728-token
logical history limit. Reuse the validated author retrieval, relative positions,
representatives, LRU/offload, Q8_0 GGUF mapping, FP8 persistent KV and HF/FLA
hybrid model adapter. Assembled attention views remain FP16. History is on CPU,
without NVMe or GC. There is no new model/kernel change for this scope switch.

Keep original tasks, tools, repository starts and hidden verifiers; no injected
history, no padding, Claude Code 2.1.148 xhigh, no client compaction or subagents.
Keep seed1000, temperature1, top-p .95, top-k20, prefill outer chunk2048,
thinking guard28672 and maximum response32000. The original task order is
SuperJSON, Textual, Koota, Awilix, as listed in the root README and task manifest.
The frozen plan's seed1000 overrides the older pilot manifest's default seed.

Current plan: `configs/infllm_small_deepswe4_s1000_20260912.json`.
Current campaign: `results/infllm_small_deepswe4_s1000_20260912/`.
Controller/report regression evidence: `reports/controller_selected_windows_20260912.json`.
Existing model validation evidence remains in the plan and was not replaced by
controller tests.

Scheduling update on 2026-09-12: the user's spare-GPU authorization is used to
run the later Awilix arm concurrently with the serial controller, at ports
18243/18244. At most two model servers are planned. Its output is the originally
planned `awilix-async-container-initialization/small/attempt_01`; it invokes the
same frozen arm harness and seed1000 plan, with no new native trials. The serial
controller skips it if already completed. If the serial controller reaches it
while it is still running, preserve both states and wait on the verified reserved
arm; resume the serial controller only after inspecting its completed result.
Never create a second attempt just because the serial incomplete-attempt guard
fires. The reservation is tracked in `parallel_arms/*.json`; the auxiliary
observer now counts actual completed result files, including out-of-order
completion. Launch and observer evidence are recorded separately. These are
shared-GPU accuracy trials with descriptive timings, not exclusive speed tests.

Report full task passes and partial hidden-test credit separately. Use complete
three-method task pairs for cohort averages. Preserve raw TTFT, decode time,
output count, total agent/request time and CPU/GPU memory. Incomplete or
interrupted runs do not become zero scores. Large-window diagnostic logs remain
under the retired `infllm_two_windows_deepswe4_s1000_20260912` campaign, with
`stopped_for_small_window_only.json`, cleanup evidence and copies of its original
controller sources under `frozen_changed_sources/`.

InfLLM currently has no MTP and uses HF/FLA BF16 intermediates; historical native
runs used MTP chain4 and FP32 residuals. GPU load is shared/partially unverified.
The small active window differs from historical KVMem's maximum 196608 tokens.
Report actual observations with these limits; do not label raw timing ratios
as pure algorithm speedups or infer large-window capability from small-window
results. Four tasks and one seed are pilot evidence.
