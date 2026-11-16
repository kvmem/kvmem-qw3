# CPU bookkeeping diagnostic for the author InfLLM implementation

This auxiliary check explains a possible implementation cost of enlarging the
active window. It is not a task evaluation, attention benchmark, observation
of live retrieval turnover, or full-model latency estimate. It did not create
a CUDA context or attach to/change any running experiment process.

The exact pinned author's `ContextManager.get_global_hidden_and_mask` ran with
inert tensor copies/events. The Python list management, slot search, copying of
block-ID lists and metadata assertions execute unchanged. GPU work and KV data
movement are omitted. The assumed replaced-block counts below are test inputs,
not measured agent behavior.

| Case | Retrieved-block capacity | Populated blocks | Assumed replaced blocks | Median metadata time per call |
|---|---:|---:|---:|---:|
| Author-small, unchanged | 16 | 16 | 0 | 0.0128 ms |
| Author-small, half replaced | 16 | 16 | 8 | 0.0203 ms |
| Large, below capacity | 1503 | 266 | 0 | 0.6117 ms |
| Large, full and unchanged | 1503 | 1503 | 0 | 6.7651 ms |
| Large, full | 1503 | 1503 | 16 | 8.1992 ms |
| Large, full | 1503 | 1503 | 150 | 102.7971 ms |

Seven independent calls per case were timed without a profiler. A separate
cProfile capture attributes the work to author metadata operations. The earlier
`author_block_bookkeeping_cpu_20260912.json` timed profiled calls and must not be
used as the unprofiled measurements; the `_v2.json` report above supersedes it.

The source searches Python lists for retained blocks, searches candidate slots
for replacements, and deep-copies the full slot-ID list twice. At high `topk`,
these operations become more expensive even though tensors are still retained
in the assembled buffer. This is a concrete implementation concern when
extending the small-window author code to a large-window decoding workload.

Do not multiply these numbers by layer count and present the result as observed
agent latency. Live overlap, block turnover, GPU work, projections, recurrence,
sampling and client work are not measured here. The live large-window trial was
below the 192K capacity at the time of this diagnostic. Actual retrieval-turnover
statistics are not available in its frozen server logs.

Any eventual native-relative speed comparison must distinguish this Python
implementation cost, backend and MTP differences, shared GPU load, and the
algorithm's retrieval frequency. More efficient equivalent data structures
could reduce this cost; it is not a theoretical lower bound for InfLLM.

Source: `upstream/InfLLM/inf_llm/attention/context_manager.py`, pinned commit
`12b70798f56e56ebb23c53c7018091a3f540a028`.
Reproducer: `profile_author_block_bookkeeping.py`.
Raw data: `reports/author_block_bookkeeping_cpu_20260912_v2.json`.
