# Q8_0 / MTP deployment rerun

User correction: use the existing Qwen3.8-27B Q8_0 weights and turn MTP on; verify rather than assume at least 50 decode tokens/s. Prior BF16/MTP=0 trials remain supplemental.

- Same frozen candidate_v4 engine and exact deployed Q8_0 GGUF (SHA256 in weight_identity.json). FP8 KV; MTP chain 4 in both main arms.
- Same exact DeepSWE prefix token IDs at 8192 / 32768 / 122880 / 245760 tokens, fixed 128 outputs, greedy, ignore EOS, three repetitions, batch one. The 1024-token warmup is excluded.
- KVMem: selection / prefill / generation budget 131072 each, block=32, 2M history, CPU history and index, no NVMe, GC off, full query replay. Requests provide an explicit current query span. The guided-reselect flag remains both for the MTP=4 arm; record whether actual private query generation occurs rather than infer it from the flag.
- Plain qw3: original 262144-token context with prefix cache / single-slot continuous batching. No client compaction in this fixed-prefix decode microbenchmark.
- Pure text: the unused vision tower is not loaded. CPU affinity is GPU-local NUMA0 (0-15,32-47). GPU measurement is exclusive; OSWorld 032/KVMem was archived as a user-interrupted attempt, not scored as a failure.
- Timing: 127 output tokens divided by first-model-token-to-server-completion interval. Initial prefill and query/replay are separate; terminal bookkeeping is included. Raw native counters, whole-request wall times and MTP draft/accept counts are all retained.
- A 32K Q8/MTP=0 KVMem control is below the selection budget and helps compare MTP against non-speculative generation on the same weight file.
- InfLLM and InfiniGen Qwen adapters do not yet implement MTP verification and hybrid-state rollback. No external row is labeled Q8/MTP=4. An optional scope question was sent; the native deployment rerun proceeds independently.
- The 240K KVMem repetitions can produce different generated text while restoring/refining historical state. Acceptance rates are reported; this is not a token-identical method speedup or an accuracy evaluation.
- One ordinary-qw3 startup hit a reused-port bind error before any inference. Logs and the initial plan are preserved in startup_failure_port_reuse/. jobs_v2.json assigns distinct ports and retains the completed KVMem arm unchanged.

After export, finish_resume.py restarts the original OSWorld suite and verifies successful API output, preserving all completed result SHA256 hashes.
