# Named KV-retrieval baseline comparison

**Superseded for future scheduling by the user's 2026-09-12 clarification:**
[long-running agent reuse plan](LONG_RUNNING_AGENT_REUSE_PLAN.md). Add only
InfLLM against the existing completed KVMem/dense agent setting. The BF16
three-condition plan below describes preserved calibration/integration evidence;
it must not automatically schedule more KVMem or dense runs.

This is a new text-only comparison. Old Q8/FP8 OSWorld/DeepSWE results are not
mixed with these BF16-weight/FP16-KV runs. All code is isolated in this directory.

## Conditions and selection

Four tasks were frozen before new outcomes: superjson-error-stack-serialization,
textual-richlog-follow-state, koota-composite-trait-aspects, and
awilix-async-container-initialization (seed 73). Original task instructions,
container worktrees, tools and hidden judges are used. There is no injected
history. The local harness permits natural completion without an external agent
wall-time deadline; this differs from task.toml's 5400-second agent limit. The
hidden verifier retains its own 30-minute limit.

The main pilot conditions are native qw3 + Claude Code auto compaction, original
InfLLM adapted to Qwen3.8, and full guided KVMem. All use Claude Code 2.1.148,
disable subagents, use the same BF16 text checkpoint and FP16 attention KV, and
sample at temperature 1, top-p .95, top-k 20. Speculative token proposals are off.
Native KVMem still maintains its required MTP prefix state and generates the
full guided query/replay. The other conditions do not load an MTP head.

The 60 GiB candidate has a 64K dense context; KVMem has 32K selection, 32K
generation reserve, 16K refresh trigger and 512K logical history; InfLLM uses
4096 local + 224 retrieved 128-token blocks, 128 initial tokens and a 256-block
GPU LRU cache, with 512K logical history. The KV methods disable client summaries.
NVMe and history GC are off. Native CPU KV backing is configured to 40 GiB.
The InfLLM attention budget is about 32K, whereas KVMem's resident window can
grow from 32K toward 64K. This is a resource-budget comparison, not equal attended
tokens at every step. Very large mandatory live spans can exceed KVMem's smaller
selection budget; such failures must be reported, not hidden by changing a task.

A second, smaller fixed-trace point uses a 58 GiB monitored process ceiling:
InfLLM has 4096 local + 160 retrieved blocks and a 192-block GPU LRU cache;
KVMem has 32K selection + 8K generation reserve, 1K refresh trigger. Parameters
were calibrated on a separate termenv transcript, not on pilot outcomes.

## Memory and timing

Account for actual cache, duplicate retrieval buffers, representative indices,
recurrent state and prefix snapshots. InfLLM's optional calibration inventory
counts unique tensor backing storages, including shared immutable CPU history.
Its CPU traversal is off for timed agent runs. PyTorch's allocator limit covers
PyTorch allocations only. Native KV pool size and fixed prefill chunk are set
explicitly. NVML samples total process memory every 2 seconds; the supervisor
stops a run if a sampled value exceeds its ceiling. This is a monitored ceiling,
not a hard native allocator limit or a guarantee that sub-second peaks were seen.

Shared-GPU runs are integration/capacity/quality evidence. Their raw times are
retained, but they do not establish formal latency ratios. Formal fixed-input
measurements require exclusive GPU ownership, including absence of queued work
that can restart between requests. Never terminate unrelated experiments.

The common proxy forwards exact request bytes and streams without waiting for a
complete reply. It archives request bytes after streaming and records first
meaningful content (including thinking), first non-thinking content, and first
tool-call start separately. Observation-to-reply timing starts at host receipt
of Claude's tool_result JSON event; stdout buffering limits its precision.
Summary requests are identified by native compact-boundary events and response
message IDs. Their first tokens are not counted as resumed actions. Unmatched
compaction boundaries must be flagged. API output token counts exclude private
retrieval queries; private-query/replay time is reported separately.

## Fidelity and interpretation

InfLLM retains the pinned author's representative-token selection, retrieval,
two-stage attention, position rule, CPU offload, LRU cache, async stream and
pinned-memory options. Adaptation is limited to Qwen's full-attention boundary,
partial rotary channels, hybrid session snapshots and kernel compatibility with
head dimension 256/SM120. Recurrent layers use the unchanged model code with FLA.

HF and native execution differ, including sampler RNG. A fixed seed does not
make their random streams identical. Dense HF with BF16 KV is an implementation
control only, not a main matched-precision baseline. Teacher-forced continuations
and separate CPU/CUDA profiles help locate overhead; intrusive profiler times
must not be used as end-to-end speedups. The observed torch conv1d operator is a
small part of current decode time; the existing optimized FLA recurrence stays
on. No unvalidated convolution replacement is used in the main pilot.

FlexiCache is audited from pinned original source. Its vLLM 0.8.2 fork lacks this
hybrid model and a Qwen3.8 head-stability profile; its default CPU pool exceeds
host RAM. A genuine comparison needs model integration and independent head
profiling. Neither borrowed head profiles nor KVMem internal ablations will be
presented as a FlexiCache result. This remains an additional-baseline feasibility
item, while the direct InfLLM comparison is the first end-to-end priority.
