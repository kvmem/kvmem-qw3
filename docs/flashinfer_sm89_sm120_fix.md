# FlashInfer prefill correctness: SM89 and SM120

Follow-up to [the initial FlashInfer investigation](issue2_sm89_flashinfer_benchmark.md)
and the native CUDA fixes in `72f4f72`. This change integrates the dependency
synchronization repair and repairs plan reuse; it does not disable FlashInfer
or turn its plan cache off by default.

## Root causes and changes

1. **Scalar shared-memory output stores need a warp barrier.** FlashInfer
   0.6.11.post2's `write_o_reg_gmem` scalar store path lets lanes gather other
   lanes' values without an intervening synchronization. CMake now prepares a
   private header overlay in the build directory, preserving the dependency's
   relative include tree and adding `__syncwarp()` only when
   `FLASHINFER_STMATRIX_M8N8X4_ENABLED` is absent. SM120 keeps its existing
   `stmatrix` path. Installed package files are never modified. An unrecognized
   or ambiguous output-gather boundary fails configuration rather than silently
   skipping the compatibility repair.
2. **A plan-cache hit still uses pinned host memory.** The adapter always
   uploads request metadata, including the last-page length. The backend used
   to skip recording the host-slot completion event on a cache hit. The
   eight-slot ring could therefore overwrite a source buffer before its
   asynchronous H2D transfer completed. Every acquired slot is now fenced,
   including partial launcher failure paths. This problem also reproduced on
   SM120, not just on SM89.
3. **Workspace addresses do not establish ownership of their contents.** The
   thread-local plan cache previously assumed that a matching device address
   still held its metadata, even after scratch reuse or allocation recycling.
   It now owns a CPU snapshot of the live plan metadata and restores it through
   the caller's fenced pinned buffer on hits. The key includes the CUDA device;
   failed/disabled planning invalidates the old entry. Small float-workspace
   fallback still works, and only the live metadata extent is copied.

## Regression coverage

- Existing FP16/FP8 paged attention oracles cover GQA=4/6 decode and GQA=6
  prefill, dimensions 128/256, partial tiles, permuted pages, output-row guards
  and three repeats against a double-precision CPU reference.
- `qw3-flashinfer-plan-cache` deliberately clears both borrowed workspaces
  between cache hits. Twelve combinations cover both dtypes and dimensions,
  no float workspace, a 16-byte fallback workspace and a 64 MiB split workspace.
- `qw3-flashinfer-decode-gqa --queued` queues 96 prefill calls with an 8K prefix
  and changing last-page lengths, without synchronization between calls.
  Both FP16 and FP8 are checked against an independent uniform-attention oracle.
  The racecheck variant `--queued-small` uses 24 calls and a 257-token prefix,
  retaining three host-ring traversals while bounding instrumentation history.
- `qw3-flashinfer-model-repeat` loads a local Q8_0 GGUF once per process,
  disables prefix reuse and MTP, and checks four identical greedy outputs after
  independent request resets. Each request generates exactly 128 tokens with
  EOS ignored. The default prompts are 6,537 and 2,185 tokens for the tested
  tokenizer, in that order to cover context-size changes.
- The model runner sweeps FP16/FP8 KV with chunks 256/1024/2048/0, plus FP16
  cache-off, FP16 native and FP32 native controls. Timings exclude the first
  request of each prompt and report the median of three measured requests.
- The host-only overlay test verifies input preservation, repeatable generation,
  the architecture guard, relative includes and rejection of unknown layouts.

The old code fails both workspace-restoration and queued-metadata regressions,
even with the earlier isolated output-sync patch. In that sync-only negative
control, workspace corruption produces a maximum error of 0.735867; the small
queued FP8 test produces 0.0390264 (tolerance 0.001). The full-model negative
control also produces different greedy outputs in all three repeats after its
first 6,537-token/chunk-1024 request on **both GPUs**.

## Completed results

| Check | SM89 | SM120 |
|---|---|---|
| FlashInfer numerical + memcheck/racecheck matrix | 8/8 pass | 8/8 pass |
| Native FP16/FP32 numerical modes | 2/2 pass | 2/2 pass |
| Full-model configurations / requests | 11/11; 88/88 pass | 11/11; 88/88 pass |
| Native attention sanitizer matrix | See preceding native campaign | 40/40 pass |
| MMQ sanitizer matrix | See preceding native campaign | 16/16 pass |

There are **176 accepted model requests** across the two GPUs, each generating
128 tokens. All four outputs in each configuration/context group match byte
for byte. FP16 cache-enabled and cache-disabled controls also match each other
for both context lengths on both GPUs. The separate host build passes **26/26**
CTest checks; final SM120 CUDA CTest passes **6/6**. Every accepted sanitizer
log reports zero errors, and accepted racecheck logs report zero warnings.

Maximum absolute errors: FlashInfer oracles including the queued cases are
0.000584722 on SM89 and 0.000557542 on SM120 (tolerance 0.001). SM120's native
attention maximum is 0.000219062; its MMQ maximum is approximately 0.00007629.

### Throughput

Median token/s for the 6,537-token input, 128-token output; one warmup and three
measured requests per configuration. FP16/FP8 below refer to KV, not weights.
All FlashInfer rows use the repaired dependency and cache; the cache remains
enabled except for the explicitly labelled control. Chunk 0 means unchunked.

| KV / attention | Chunk | SM89 prefill | SM89 decode | SM120 prefill | SM120 decode |
|---|---:|---:|---:|---:|---:|
| FP16 / FlashInfer | 256 | 2190.23 | 32.48 | 2770.76 | 46.32 |
| FP16 / FlashInfer | 1024 | 2687.72 | 32.48 | 4119.82 | 46.31 |
| FP16 / FlashInfer | 2048 | 2761.19 | 32.46 | 4316.26 | 46.30 |
| FP16 / FlashInfer | 0 | 2685.88 | 32.47 | 4449.83 | 46.30 |
| FP8 / FlashInfer | 256 | 2159.78 | 32.52 | 2758.81 | 46.52 |
| FP8 / FlashInfer | 1024 | 2659.00 | 32.54 | 4085.09 | 46.53 |
| FP8 / FlashInfer | 2048 | 2735.21 | 32.51 | 4290.29 | 46.54 |
| FP8 / FlashInfer | 0 | 2661.18 | 32.52 | 4415.85 | 46.46 |
| FP16 / FlashInfer, cache off | 1024 | 2682.87 | 32.47 | 4109.87 | 46.31 |
| FP16 / native | 1024 | 2636.41 | 31.51 | 3992.66 | 43.29 |
| FP32 / native | 1024 | 1248.42 | 31.40 | 1883.94 | 43.07 |

A separate SM120 before/after pair uses matching `-O3 -DNDEBUG -lineinfo`
SM120a builds and the same FP16/chunk-1024 workload. Only the prefill adapter,
dependency overlay and backend host-slot protection differ; the previously
committed native fixes remain present in both. Prefill is **4099.61 before vs
4114.74 after**, and decode is **46.32 vs 46.32**. The old version again has
three mismatching repeats; all four fixed outputs match. This supports no
material throughput regression in this workload, not a general speedup claim
or a before/after performance measurement of the earlier native-kernel fixes.

### Serving implications

For these validated Q8_0 + FP16/FP8 KV serialized paths, FlashInfer can remain
enabled after rebuilding. There is no longer a need to set
`QW3_FLASHINFER_PLAN_CACHE=0` or force native attention as a workaround for these
bugs. Remove any prior `QW3_PREFILL_ATTN=mma-gqa-v2` /
`QW3_DECODE_ATTN=native` override if the intention is to use FlashInfer.
The service CLI/API is unchanged. Keep the architecture appropriate to the
deployment GPU; do not copy an SM89-only binary as an SM120 build or vice versa.
The existing production build and running services were not replaced/restarted.

The native fixes in `72f4f72` affect shared CUDA code and can therefore affect
SM120 too. The native oracle/sanitizer matrix above found no numerical or
memory-access regression on SM120. The new scalar-output barrier is excluded
from its `stmatrix` branch, while the host-metadata fixes deliberately apply to
both architectures because the old metadata bug reproduced on both.

## Reproduction

Configure with CUDA, FlashInfer ON, `QW3_BUILD_TESTS=ON`, Release and `-lineinfo`.
Use architecture `89` for the tested 4090 and `120a` for the tested Blackwell.
Set `QW3_FLASHINFER_INCLUDE_DIR` to the installed dependency's `data/include`.
This campaign uses FlashInfer **0.6.11.post2**, CUDA **13.0.88**, GCC **11.4**.
The original prefill-header SHA256 is
`ed83f5964cf1d815f80393248360ae525bdf64fb53bc1bfaf58bf03d42b9b464`;
the build-private patched header SHA256 is
`bfa320507712c00485600de24526a176edbf1f863897d4e64233de90732e0535`.

```bash
cmake --build BUILD --target qw3 qw3-flashinfer-decode-gqa \
  qw3-flashinfer-plan-cache qw3-flashinfer-model-repeat \
  qw3-native-attention-oracle qw3-mmq-parity -j4
ctest --test-dir BUILD -L 'flashinfer_attention|native_attention' --output-on-failure
bash tests/run_flashinfer_sanitizers.sh BUILD LOGS/flashinfer
bash tests/run_native_attention_sanitizers.sh BUILD/qw3-native-attention-oracle LOGS/native
bash tests/run_mmq_sanitizers.sh BUILD/qw3-mmq-parity LOGS/mmq
bash tests/run_flashinfer_model_regressions.sh BUILD /path/to/model.gguf LOGS/model
```

Set `COMPUTE_SANITIZER` and `LD_LIBRARY_PATH` if the toolkit is not on the default
search paths. Run on an idle GPU; do not mix sanitizer runs with timing runs.
Model weights are never downloaded automatically by these tests.

## Hardware and limits

| GPU | Architecture | VRAM reported | Driver |
|---|---|---|---|
| RTX 4090, 48 GB configuration | SM89 | 49,140 MiB | 580.105.08 |
| RTX PRO 6000 Blackwell Server Edition | SM120a build | 97,887 MiB | 580.159.03 |

Both use the same Qwen3.8-27B-Q8_0 GGUF, SHA256
`a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`.
Builds include the existing unrelated working-tree changes, which are preserved.

The original 8K/96-call **racecheck** stress attempts are not accepted passes:
SM89 timed out after 300 seconds; the SM120 target exited abnormally (exit 9)
without a successful test completion. The full stress case remains in normal
CTest and memcheck; accepted racecheck coverage uses the bounded 257/24 case.
Do not interpret this as a full-model or all-workload racecheck pass.

This validates the tested Q8_0 weight path, not NVFP4/FP8 weights, MTP, vision,
concurrent serving, KVMem, WSL2, SM86 or fitting a 27B Q8_0 model on a stock
24 GB 4090. SM120 NVFP4 sources are compiled, but compilation alone is not an
NVFP4 numerical or performance acceptance test.

Commands, statuses, raw logs, per-context summaries and source/binary hashes are
retained locally in `benchmark/issue2_ada_20260919/flashinfer-fix-results/`.
`summarize.py` requires every expected configuration, checks exactly 128 decoded
tokens per request, repetition completeness and byte equality, and compares
cache-on/off outputs. `manifest.py` checks accepted sanitizer summaries before
producing the inventory. This artifact directory is ignored by Git; the tests
and this report are source files. The original validation campaign did not
push changes, update the public issue, or replace/restart production services.
