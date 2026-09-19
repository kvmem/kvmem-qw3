# Issue #2: native Q8_0 / FP16 KV on SM89

This change addresses the failures reproduced on RTX 4090 with FlashInfer OFF.
It retains native MMA attention as the default; it does not merely select `vec`.

The subsequent FlashInfer-ON compatibility/throughput campaign is documented
separately in [the SM89 FlashInfer report](issue2_sm89_flashinfer_benchmark.md).

## Changes

- Match the seven 32-bit shared-address `ldmatrix` operands with explicit
  `.shared` instructions. Leave the valid 64-bit generic-address MMQ form intact.
- Synchronize warp readers before lane 0 overwrites online-softmax state.
- Synchronize all block readers before single-buffered K/V storage is reused by
  the next tile. Apply the same repair to the related MMA attention variants.
- Restore the architecture guard in the separate legacy-kernel translation unit;
  otherwise its v3/v4/v5 MMA helpers silently compile to no-ops.
- Restrict v2's forced split count to its instantiated values (1/2). Unsupported
  overrides such as 4 use the normal heuristic, keeping scratch/grid/kernel
  layouts consistent. v1's supported four-split kernel is unchanged.
- Keep plain local caches contiguous when FlashInfer is absent, native decode
  is explicitly selected, or the KV dtype is unsupported by FlashInfer decode.
  Explicit external paged/KVMem paths are not reinterpreted as contiguous caches.
- Detect the cuBLASLt outer-scale API at configure time, so CUDA 12.4 can build
  Q8/FP16 without those newer FP8 symbols. Unsupported outer scaling returns an
  explicit error rather than silently omitting scaling.

## Regression checks

Build with CUDA, SM89, FlashInfer OFF and `QW3_BUILD_TESTS=ON`.

```bash
cmake -S . -B build-sm89 -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON -DQW3_ENABLE_FLASHINFER=OFF \
  -DQW3_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build build-sm89 --target qw3 qw3-native-attention-oracle qw3-mmq-parity -j6
ctest --test-dir build-sm89 -L native_attention --output-on-failure
bash tests/run_native_attention_sanitizers.sh \
  build-sm89/qw3-native-attention-oracle build-sm89/sanitizer-logs
```

The attention oracle uses deterministic random inputs, GQA=6, dimensions 128/256,
non-aligned query lengths, prefix lengths 0/31/63/127/257/997, masked rows/empty KV
splits, sharper softmax inputs, and output-row sentinels. Each of seven GPU cases
repeats three times against a double-precision CPU reference. Absolute tolerances are 0.001 for FP16 MMA and
0.00001 for FP32. `QW3_PREFILL_ATTN_MIN_BATCH=1` ensures even the small cases
actually enter the selected native kernel.

The sanitizer matrix runs separate processes for tile/split/padding choices,
both asynchronous-copy toggles, the FP16 accumulator option, the base/pipelined
MMA kernels, GQA variants v1–v5, and the vec control. It finishes with FP32 KV
under both sanitizers: 38 FP16 runs plus 2 FP32 runs.

For integer MMQ coverage:

```bash
bash tests/run_mmq_sanitizers.sh build-sm89/qw3-mmq-parity build-sm89/mmq-logs
```

This exercises versions 2–8 and auto under both memcheck and racecheck. `--small`
bounds the CPU oracle cost, retains multi-K-tile cases, and adds row/batch tails.
The oracle stages the matching 36-byte or 144-byte activation format for each
kernel, rejects non-finite output, and fails if no CUDA device is available.

Full-model acceptance additionally requires the issue's exact prompt/options
without workaround variables, FP32 KV, and prompts longer than a 1024-token
prefill chunk. A successful single token or memcheck alone is insufficient:
the earlier address-only fix passed memcheck but failed racecheck and numerical
parity.

## Results

Hardware: provided RTX 4090 **48 GB configuration**, SM89, driver 580.105.08;
Ubuntu 22.04.4 container, GCC 11.4.0. All GPU tests run serially with clean
dispatch environments, FlashInfer OFF, Release and `-lineinfo`.
Both builds are checked with the task-local CUDA 13 Compute Sanitizer.

| Check | CUDA 13.0.88 | CUDA 12.4.131 |
|---|---|---|
| Build CLI and both GPU oracles | Pass | Pass |
| Attention numerical oracle + memcheck/racecheck | 40/40 pass | 40/40 pass |
| MMQ numerical oracle + memcheck/racecheck (9 shapes/run) | 16/16 pass | 16/16 pass |
| Full-model acceptance probes | 11/11 pass | 5/5 pass |

The separate host-only build passes **25/25 CTest tests**; CUDA 12.4 also passes
both registered native-attention CTest tests. CUDA 12.4 correctly
detects that outer-scale enums are absent; the feature probe's negative result
is expected and does not prevent building Q8/FP16.

Before committing, the staged sources were exported without unrelated worktree
changes and independently built in Release mode with CUDA OFF. That scoped
snapshot builds the CLI and passes **21/21 host CTest tests**, plus shell syntax
checks for both sanitizer scripts. Its smaller test count excludes four
uncommitted KVMem tests. GPU results above remain results of the earlier working
tree campaign, not a second GPU campaign on the scoped snapshot.

Attention's largest FP16 absolute error is **0.000219181**, below 0.001;
the default MMA-v2 maximum is **0.0000810474**. FP32's maximum is
**0.0000000596046** (tolerance 0.00001). MMQ's maximum absolute difference is
**0.00007629**. These maxima match across the two builds. All accepted
memcheck runs report zero errors; all accepted racecheck runs report zero
errors and zero warnings.
The new split-layout assertion rejects the pre-guard binary with
`QW3_PREFILL_FA2_NSPLIT=4`, verifying that this regression check detects the bug.

The CUDA 13 full-model checks use the same 27B Q8_0 GGUF as the original A/B
diagnosis, SHA256:

`a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`

- Original prompt/options, FP16 KV: 132 generated tokens, normal EOS, exit 0.
- Same command, FP32 KV: 128 generated tokens, normal EOS, exit 0.
- 2208-token prompt with chunk=1024, chunk=256, and chunking disabled:
  each generates 32 tokens, exit 0.
- 6560-token prompt with chunk=1024: 32 generated tokens, exit 0.
- `--no-paged-kv` and `QW3_PAGED_KV_PREFILL=1`: both complete normally, exit 0.
- Full-model memcheck, 8 generated tokens, FP16 and FP32: zero errors.
- Full-model attention-filtered racecheck, prefill plus 8 generated tokens:
  zero errors and zero warnings.

CUDA 12.4 independently passes the default FP16 command, FP32 command,
2208-token/chunk=1024 prefill, full-model FP16 memcheck and attention-filtered
racecheck. Its displayed output matches CUDA 13 for all three generation probes
(132/128/32 generated tokens respectively).

No `vec`/paged-cache workaround is needed for the default command:

```bash
build-sm89/qw3 --model /path/to/Qwen3.8-27B-Q8_0.gguf \
  --ctx 8192 --kv-dtype fp16 --prefill-chunk 1024 \
  --temp 0 --top-p 1 --top-k 1 \
  -p 'Explain virtual memory in one paragraph.' -n 256
```

## Serving after rebuilding

The `qw3 serve` CLI and HTTP API are unchanged. Rebuild the executable to pick
up the fixes; restarting an old binary is not sufficient. For the validated
SM89 native path, use the FlashInfer-OFF build above and ordinary serialized
serving:

```bash
build-sm89/qw3 serve --model /path/to/Qwen3.8-27B-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 --ctx 8192 \
  --kv-dtype fp16 --prefill-chunk 2048 --mtp-chain 0 \
  --no-continuous-batching --no-paged-kv
```

Chunk size 2048 is already the server default and was the fastest native
prefill choice in the measured 6,537-token sweep; it is not required for
correctness. Adjust the context and model to available VRAM. The tested 27B
Q8_0 model used a 48 GB 4090, not a stock 24 GB card.

If reusing a FlashInfer-ON build for this same plain FP16 path, explicitly set
both `QW3_PREFILL_ATTN=mma-gqa-v2` and `QW3_DECODE_ATTN=native` before the command
to select native attention. This is not a substitute for FlashInfer on external
paged/KVMem paths. Continuous batching, prefix-cache and KVMem serving are not
covered by this native validation. The FlashInfer report documents outstanding
prefill synchronization and plan-reuse issues; neither its isolated dependency
patch nor the plan-cache workaround is integrated by this change.

## Coverage limits and artifacts

- Full-model **unfiltered** racecheck, and a broader fattn+MMQ-filtered run,
  hit the 240-second budget on the pre-split-guard binary. Neither is counted
  as a pass. Full-model racecheck acceptance is restricted to `fattn` kernels;
  MMQ is independently checked on all nine oracle shapes for versions 2–8/auto.
  This is not an all-kernel/full-model racecheck pass.
- The reporter's exact WSL2, toolkit/driver and model-file combination was not
  available. This verifies the reproduced SM89 failure mechanisms in native
  Linux, not that exact installation or a standard 24 GB 4090 model fit.
- No A40/SM86, FlashInfer ON, external paged/KVMem, FP8 or MTP GPU validation
  is claimed. This does not change the broader release support matrix.
- The GPU campaign used the existing `codex/demo` working tree plus the fix,
  not a clean HEAD build. Existing user changes were preserved. The subsequent
  fix commit excludes unrelated worktree changes; no push or issue update was
  performed.

Binary/source hashes are in `benchmark/issue2_ada_20260919/fix-manifest.json`;
commands, statuses and raw logs are in its `fix-results/` directory. The selected
full-model probes are reproducible with `remote-probe.sh` in the parent directory.

The original diagnostic artifacts remain in `benchmark/issue2_ada_20260919/`.
Their partial-fix warnings describe the earlier experiment, not this final change.
