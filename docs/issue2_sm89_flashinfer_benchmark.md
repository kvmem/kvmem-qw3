# SM89 FlashInfer compatibility and single-request throughput

This is a follow-up to `issue2_sm89_fix.md`, using the repaired working tree on
the same remote RTX 4090. Results below apply to this measured configuration,
not all FlashInfer operations or all supported GPU architectures.

## Compatibility findings

FlashInfer 0.6.11.post2 builds and runs Q8_0 model inference on SM89 with both
FP16 and FP8 KV. The ordinary 31-token CLI prompt finishes normally at EOS:
122 generated tokens with FP16 and 127 with FP8. FP32 KV also works (128 tokens)
but uses native attention, not a FlashInfer FP32 implementation.

The stock dependency is **not a clean sanitizer pass**:

- Paged decode: six FP16/FP8 GQA=4/6 shapes, three repetitions each, pass the
  numerical oracle, memcheck and racecheck; maximum absolute error 2.83611e-6.
- Paged prefill: twelve FP16/FP8 shapes, three repetitions each, pass the
  numerical oracle and memcheck; maximum absolute error **0.000116721**, below
  the 0.001 tolerance. FP8 is compared with the actual quantized FP8 cache;
  this is not an FP8-versus-FP16 model-quality evaluation.
- Stock prefill racecheck exits **99**, reporting **192 warnings** (zero
  error-severity findings). Warning severity is not grounds to ignore missing
  warp synchronization on SM89. See NVIDIA's
  [Racecheck documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
- The separate contiguous FP16 and native FP32 fallback oracles pass.
- Stock FlashInfer's longer greedy requests are not byte-identical across
  repetitions, including text before the first EOS. The native baseline is
  repeatable for the tested prompts. This alone does not measure task quality.

### Isolated synchronization experiment

The reported accesses are in FlashInfer's `write_o_reg_gmem`, where SM89's
non-`stmatrix` branch writes output fragments into shared memory and then other
lanes read them for vectorized global stores without an explicit warp barrier.

An **isolated header copy**, not the installed dependency or repository runtime,
adds `__syncwarp()` between those stores and reads. Only the prefill adapter was
recompiled with that copy, then substituted into a copy of the built static
library. Binaries are explicitly labeled `fi-sync`; stock `fi` binaries and
results are retained separately. This is a tested diagnostic patch, not an
integrated dependency update, and does not claim vision/NVFP4/all-architecture
coverage.

For a bounded model-shaped case (head dimension 256, GQA=6, batch=31), each stock
FP16/FP8 run reports 48 warnings; each patched run reports **zero errors and zero
warnings**, with unchanged numerical error. The patched complete twelve-shape
prefill matrix also passes both memcheck and racecheck with zero findings.
However, the patch **does not eliminate all full-model repeated-output
differences**: synchronization warnings and generation repeatability must not be
conflated or advertised as one completely resolved problem.

At 6,537 input tokens, the patched FlashInfer **contiguous** KV control
(`QW3_PAGED_KV_PREFILL=0`) is byte-identical across all four requests, including
warmup. Its medians are 2,677.01 prefill token/s and 32.47 decode token/s. The
FlashInfer-enabled binary with explicit native prefill/decode is also repeatable
and matches the native baseline's displayed output. This narrows the observed
repeatability issue to the tested paged FlashInfer route; it does not yet prove
its precise cause or a general quality failure.

Two further paged controls use the same patched binary, FP16, chunk=1024,
6,537 input tokens and four requests:

| Control | Prefill token/s | Decode token/s | All four outputs identical |
|---|---:|---:|---|
| Default plan cache | 2,668.16 | 32.43 | No |
| `QW3_FLASHINFER_PLAN_CACHE=0` | 2,686.32 | 32.48 | **Yes** |
| `QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=0` (no KV splitting) | 2,678.34 | 32.44 | No |

The plan-cache-off output matches the contiguous control byte-for-byte. Disabling
KV splitting alone does not restore repeatability. This implicates **QW3's
prefill plan reuse**, separately from FlashInfer's shared-memory barrier issue;
the precise invalidation/metadata-lifetime mechanism is not yet established.
Do not infer that adding the barrier alone fixes both problems.

## Configuration and method

- GPU: NVIDIA GeForce RTX 4090, **48 GB configuration**, SM89, driver 580.105.08,
  450 W power limit. No other GPU workload ran during the measurements.
- Model: the same 27B Q8_0 GGUF used for the issue reproduction;
  SHA256 `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`.
- CUDA 13.0.88, GCC 11.4.0, Release, `-lineinfo`, CUDA architecture `89`.
- FlashInfer headers: **0.6.11.post2**, including that installation's bundled
  CCCL. The original prefill header matches its installed wheel RECORD hash.
  No Python/PyTorch inference or FlashInfer Python JIT is involved.
- Native and FlashInfer builds use the same repaired runtime sources. This is
  the existing dirty `codex/demo` working tree, not a clean release/HEAD build.
- One request at a time, context capacity 24,576, MTP disabled, prefix reuse
  disabled, greedy sampling, exactly 128 output tokens (`ignore_eos=true`).
- Each process loads the model once. Each prompt/configuration runs once for
  warmup, then three measured repetitions; every request resets KV/recurrent
  state. Tables use median token/s from the backend's synchronized phase timers.
  Loading and sanitizer overhead are excluded. Raw logs retain warmup timings.
- Benchmark prompts are plain text, **without CLI chat wrapping**: 8, 2,185,
  6,537 and 17,417 input tokens. Longer prompts repeat the same virtual-memory
  sentence before a question. This is a controlled throughput workload, not a
  model-quality benchmark. Forced continuation after EOS is intentional here;
  separate CLI compatibility smokes retain ordinary EOS behavior.
- GPU memory is sampled once per second with `nvidia-smi`; reported values are
  sampled **whole-process peaks**, not exact allocation peaks or KV-only sizes.
  The chunk=1024 multi-length process shares one peak across its four lengths.

## Throughput

All rates below are **prefill / decode token/s**, median of three measured
requests, each generating exactly 128 tokens after a separate warmup. `FI-sync`
means the isolated synchronization patch above; its default paged path still
has the documented repeatability limitation. These are throughput observations,
not a claim that the paged path has passed full correctness acceptance.

### Prefill chunk sweep: 6,537 input tokens

| Prefill chunk | Native FP16 | FI-sync FP16 | FI-sync FP8 |
|---|---:|---:|---:|
| 256 | 2,136.60 / 31.50 | 2,182.80 / 32.43 | 2,164.10 / 32.54 |
| 1024 | 2,645.54 / 31.47 | 2,668.16 / 32.43 | 2,628.29 / 32.53 |
| 2048 | 2,715.34 / 31.50 | 2,749.72 / 32.46 | 2,721.11 / 32.50 |
| Whole prompt (0) | 2,641.99 / 31.51 | 2,691.24 / 32.44 | 2,667.12 / 32.51 |

Chunk 2048 is the best measured prefill setting here. At that chunk, FI-sync
FP16 improves prefill by about **1.3%** and decode by **3.0%** over native FP16.
FP8 does not offer a material throughput improvement over FP16 in this workload.
FP32 KV is much slower for prefill: native 1,249.49 / 31.42; FlashInfer-enabled
native fallback 1,248.18 / 31.44, both at chunk 1024.

### Context-length sweep: chunk 1024

| Input tokens | Native FP16 | FI-sync FP16 | FI-sync FP8 |
|---|---:|---:|---:|
| 2,185 | 2,625.03 / 32.45 | 2,626.81 / 32.57 | 2,600.39 / 32.56 |
| 6,537 | 2,645.54 / 31.47 | 2,668.16 / 32.43 | 2,628.29 / 32.53 |
| 17,417 | 2,478.73 / 30.83 | 2,584.85 / 31.64 | 2,520.04 / 31.88 |

At 17,417 input tokens FI-sync FP16 improves prefill by about 4.3% and decode by
2.6%. This does not imply the same gains at 128K or with concurrent requests.

### Sampled whole-process peak GPU memory

| Chunk | Native FP16 (MiB) | FI-sync FP16 (MiB) | FI-sync FP8 (MiB) |
|---|---:|---:|---:|
| 256 | 29,925 | 29,999 | 29,231 |
| 1024, multi-length process | 30,197 | 30,285 | 29,517 |
| 2048 | 30,539 | 30,567 | 29,799 |
| Whole prompt | 32,053 | 32,209 | 31,441 |

At fixed context capacity, FP8 saves about **768 MiB (0.75 GiB)** versus FP16 in
these FI runs, not half the total memory. The model's Q8_0 weights are unchanged
and dominate the allocation. Whole-prompt prefill costs about 1.6 GiB more than
chunk 2048 without improving throughput here.

## Reproduction and artifacts

The completed sweep contains **35 configuration/context combinations**: one
warmup plus three measured requests each, **140 model requests / 105 measured
requests**, all exiting successfully with exactly 128 generated tokens. That
execution-success count does not override the sanitizer/repeatability findings.

Build the FlashInfer version with the existing CMake integration:

```bash
cmake -S . -B build-sm89-fi -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON -DQW3_ENABLE_FLASHINFER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=-lineinfo \
  -DQW3_FLASHINFER_INCLUDE_DIR=/path/to/flashinfer/data/include
cmake --build build-sm89-fi --target qw3 qw3-flashinfer-decode-gqa \
  qw3-native-attention-oracle -j4
ctest --test-dir build-sm89-fi -L flashinfer_attention --output-on-failure
```

The extended `qw3-flashinfer-decode-gqa` test also accepts `--prefill`. It checks
FP16 and FP8 against an independent double-precision causal attention/gating
reference, including permuted physical pages, row guards, repeated launches,
GQA=4/6 decode, and GQA=6 prefill with head dimensions 128/256 and partial tiles.
The new prefill mode is registered with CTest.

Local artifacts are under `benchmark/issue2_ada_20260919/`:

- `flashinfer_benchmark.cpp`: load-once/reset-per-request benchmark using Engine.
- `run-flashinfer-benchmark.sh`: clean-process parameter selection and GPU sampling.
- `run-flashinfer-validation.sh`: numerical/sanitizer and ordinary CLI checks.
- `run-flashinfer-sync-diagnostic.sh`: bounded stock/patched racecheck A/B and
  complete patched prefill checks.
- `flashinfer-results/diagnostic/flashinfer-output-sync.patch`: isolated header
  experiment; it is **not applied** to the installed FlashInfer dependency.
- `summarize-flashinfer.py`: validates token counts/completeness and produces CSVs.
- `flashinfer-results/`: raw commands, logs, statuses, memory samples, CSV summaries.
- `flashinfer-manifest.json`: source, binary, dependency-header and result hashes,
  original failure statuses and isolated diagnostic results.

## Recommended next steps and limits

- For the current unmodified dependency/runtime, keep **native attention + FP16
  KV + chunk=2048** as the conservative setting on this server.
- To enable the tested FlashInfer path responsibly, integrate the pinned
  dependency's output synchronization repair and disable QW3's prefill plan cache
  until its workspace/metadata ownership is corrected and covered by a repeated
  multi-chunk model regression. The measured cache-off control showed no material
  performance loss. This campaign records a diagnostic patch and tests, not that
  production integration.
- The combination of the synchronization patch and plan-cache-off was checked
  for FP16 at 6,537 tokens. Do not extrapolate that particular workaround's
  repeatability result to FP8, every context size, concurrent serving or KVMem.
- No full-model sanitizer campaign for FlashInfer, WSL2, A40/SM86, stock 24 GB
  4090 model fit, CUDA 12.4 FlashInfer build, MTP, vision, NVFP4, external paged KV,
  long-context task quality or 128K performance is claimed.
- FP16 and FP8 are **KV-cache** choices here; model weights remain Q8_0. Kernel
  oracle parity and a coherent paragraph do not establish general model-quality
  equivalence between those dtypes or between attention implementations.

The benchmark artifact directory is locally retained but ignored by Git; it is
not included in the subsequent fix/test/report commit. No push, public issue
comment, or release support-matrix change is implied.
