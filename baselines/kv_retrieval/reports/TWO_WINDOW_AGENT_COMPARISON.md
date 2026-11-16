> Historical protocol, superseded by the user on 2026-09-12. Large-window testing was canceled; follow [SMALL_WINDOW_AGENT_COMPARISON.md](SMALL_WINDOW_AGENT_COMPARISON.md). Original changed controller sources are preserved in the retired campaign.

# InfLLM active-window comparison in the existing agent setting

User-authorized on 2026-09-12. Add two InfLLM variants to the original first
four DeepSWE tasks at seed 1000; reuse completed KVMem and ordinary qw3 results.
Do not rerun those methods or substitute long-document QA for live agent work.

| Setting | Large InfLLM | Author-small InfLLM |
|---|---:|---:|
| Local tokens | 4096 | 4096 |
| Initial/sink tokens | 128 | 128 |
| Retrieved blocks | 1503 | 16 |
| Tokens per block | 128 | 128 |
| Total active token ceiling | 196608 | 6272 |
| GPU cached blocks | 3006 | 32 |
| Logical context capacity | 3145728 | 3145728 |
| Weights | Existing Q8_0 GGUF | Same |
| Persistent KV | Raw FP8 E4M3FN, scale=1 | Same |
| Attention computation/assembly | FP16 | Same |
| Prefill outer chunk / author execution chunk | 2048 / 512 | Same |
| Sampling | temperature 1, top-p .95, top-k 20, seed 1000 | Same |
| Thinking guard / maximum response | 28672 / 32000 | Same |
| Claude Code | 2.1.148, xhigh, original archived agent script | Same |
| Client compaction / subagents | Off / off | Same |
| Historical KV backing | CPU, no NVMe | Same |

KVMem has a 131072 selection budget and 65536 generation budget. Matching the
196608-token capacity ceiling avoids giving InfLLM a smaller maximum visible
history. KVMem's working set grows between refreshes; a fixed InfLLM window is
not an exact match to its instantaneous active tokens. Report this distinction.

The author's Qwen configuration is `upstream/InfLLM/config/qwen-inf-llm.yaml`:
4096 local + 16*128 retrieved + 128 sink. Both variants use the author's fused
attention and asynchronous execution options. Its representative selection,
per-layer retrieval, positional rules, LRU and CPU offloading remain in the
pinned source. FP8 storage is an explicitly validated adaptation.

## Cohort and outcomes

1. superjson-error-stack-serialization
2. textual-richlog-follow-state
3. koota-composite-trait-aspects
4. awilix-async-container-initialization

Use the original instruction, repository base commit, tool environment and
hidden verifier. No injected history or padding. Histories grow from live tool
observations and model generations. Alternate variant order across tasks;
preserve all attempts and inspect infrastructure failures before any retry.
The old BF16/32K/seed73 pilot was explicitly retired and is not comparable data.

Both variants report task reward, completed model/tool requests, observation to
first content/action wait, API TTFT, decode time/token, total agent/request time,
maximum logical prompt and GPU/CPU memory. First cold starts and later cold
starts remain visible. A zero task reward is a valid completed trial; a tool,
server or verifier infrastructure error is separately identified.

## Validation and limits

- GGUF SHA256 matches the existing run's locked model hash exactly.
- Nine recurrent tensor mapping checks against source safetensors pass; Q8
  blocks undergo only lossless head/layout permutations.
- Author FP16 attention on already-quantized inputs and adapted FP8 storage
  give identical retrieval IDs and outputs in the exercised retrieval fixture.
- Real 27B Q8 model produces coherent output and the same short-prompt argmax
  as the same quantized HF dense model. This is not native bitwise parity.
- Full large-window buffers plus tail rewrite restore fit in 50.692 GiB peak
  Torch reservation, after removing the obsolete branch's duplicate cache.
- Hybrid FP8 history/checkpoint recovery passes two fixtures. A separate
  202752-token synthetic component probe retrieves 1503 of 1551 historical
  blocks; this is correctness evidence, never a scored agent task.

InfLLM currently uses HF/FLA BF16 intermediate activations and no MTP. Existing
native qw3/KVMem used an FP32 residual stream and MTP chain4. Numeric seeds are
equal but the sampling RNG implementations differ. Shared-GPU timings are
diagnostic and cannot support formal speedup ratios against historical runs
whose concurrent GPU load is not fully verified. The ordinary qw3 batch
manifest explicitly permitted nonexclusive GPU use.
The two InfLLM variants share the same implementation, but their live task
trajectories can differ. Four tasks and one seed are pilot evidence.

`gpu_process_utilization.jsonl` additionally records per-process `nvidia-smi
pmon` samples every minute, starting partway through the first large-window
trial. Missing earlier observations are unknown. SM activity percentages are
diagnostic samples, not kernel occupancy or a factor for correcting latency.
This observer runs separately; all 44 frozen experiment sources remain intact.

Decode throughput is output tokens divided by recorded decode seconds, rather
than an unweighted mean of request throughputs. Native decode seconds come from
the backend log; the InfLLM timer spans the sampling/model/streaming loop. The
InfLLM-only `token_frontend_s` field measures token detokenization/protocol
bridging and emission within that loop. Its historical native counterpart is
unknown, not zero. The first six completed requests total 326.369 decode
seconds and 0.459 frontend seconds (0.141%); this rules out that measured
frontend portion as the main cost in those requests. The remaining time still
mixes model execution, Python scheduling, retrieval and shared-GPU waiting.

Frozen plan: `configs/infllm_two_windows_deepswe4_s1000_20260912.json`.
Raw campaign: `results/infllm_two_windows_deepswe4_s1000_20260912/`.
Validation: `reports/aligned_q8_fp8_validation_20260912_v2.json` and
`reports/fp8_long_window_components_20260912.json`.
