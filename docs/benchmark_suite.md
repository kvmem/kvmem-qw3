# Benchmark Suite

Three independent commands for smoke, efficiency, and accuracy testing, plus an
optional OpenHands SWE-bench workflow for coding-agent evaluation.

## Setup

```sh
cp benchmark/manifest.env.example benchmark/manifest.env
# Edit MODEL, LLAMA_COMPLETION, and paths as needed.
```

## Commands

| Command | When to run | ~Duration |
|---------|-------------|-----------|
| `./scripts/run_benchmark_smoke.sh` | After each code change | ~30 min |
| `./scripts/run_benchmark_efficiency.sh` | Efficiency validation / release | ~4–8 h |
| `./scripts/run_benchmark_accuracy.sh` | Accuracy validation / release | ~4–8 h |
| `python3 scripts/run_openhands_swebench.py ...` | Agent/SWE-bench validation | variable |

Results land under `benchmark/results/<git-sha>_<timestamp>/{smoke,efficiency,accuracy}/`.

## Smoke

- `ctest`
- `paged_kv_regression.py`
- `continuous_batching_regression.py`
- `mtp_continuous_regression.py`
- `continuous_prefill_interleave_regression.py`
- `kvmem_e2e_regression.py`
- Short `continuous_batching_benchmark.py` + `mtp_throughput_benchmark.py`

`kvmem_e2e_regression.py` is model-gated and runs through the real `qw3`
binary/server. It checks KVMem identity equivalence against plain decode,
step-update identity equivalence, sparse visibility engagement on a codeword
prompt, single-request KVMem + MTP, KVMem + continuous batching, and the explicit
hard-error guard for KVMem + continuous batching + MTP. Set
`SKIP_KVMEM_E2E=1` to skip it in smoke runs when only the model-free CTest suite
is desired.

The runner also accepts `--kvmem-cpu-bytes`, `--kvmem-nvme-dir`, and
`--kvmem-nvme-bytes`. When these are set, it enables `QW3_KVMEM_TIER_TRACE=1`
and adds retrieval-stage-in plus bounded single-request GPU-pool cases; the run
fails unless sparse KVMem emits stage-out activity, retrieval emits stage-in
activity, and the bounded-pool case emits `bounded_gpu_pool` while completing a
long prefill with CPU/NVMe offload.

## Efficiency

- `long_prompt_sweep.py` vs llama.cpp (default + paged KV)
- Full continuous batching matrix
- MTP throughput at 4K + long-context spots (32K–128K)
- FP8 KV spot check
- `summary_efficiency.md`

## Accuracy

Requires `lm_eval` in `.venv` (see [docs/lm_eval_harness.md](lm_eval_harness.md)).

- Production serve: continuous batching + paged KV
- Full official local suite (`benchmark/qwen36_official_local.yaml`)
- GSM8K under production / no-MTP / MTP serve configs
- `summary_accuracy.md`

Freeze baselines under `benchmark/baselines/lm_eval/` for regression diffs.

## OpenHands SWE-bench

Requires an already running OpenAI-compatible `qw3 serve` endpoint and an
OpenHands/benchmarks checkout. See [docs/openhands_swebench.md](openhands_swebench.md).

- Headless OpenHands rollout through `swebench-infer`
- Optional OpenHands `LLMSummarizingCondenser` compact baseline
- Conversion to SWE-bench `predictions.jsonl`
- Official SWE-bench scoring through `swebench-eval`
