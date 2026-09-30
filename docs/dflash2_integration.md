# Native MTP and DFlash 2 selection

DFlash 2 is integrated into the main QW3 source tree as an alternative
speculative decoder. Target execution, target verification, sampling and KVMem
remain in the native runtime. The target's built-in MTP head and the independent
DFlash checkpoint are mutually exclusive.

## Selecting a decoder

| CLI | Behavior |
| --- | --- |
| `--speculative-method none` | Plain target decoding; also suppresses legacy MTP environment activation. |
| `--speculative-method mtp --mtp-chain 4` | Built-in native MTP. Omitting the chain with explicit `mtp` selects 4. Requires matching MTP weights in the target. |
| `--speculative-method dflash2 --dflash-model DIR` | Independent DFlash 2 draft, sharing the target embedding and LM head. Does not load the native MTP head. |
| No method argument | Legacy behavior: serving defaults to no speculation; `--mtp-chain N` enables native MTP serving, and `--dflash-model DIR` selects DFlash. |

The method applies to both one-shot generation and `serve`. Explicit contradictory
flags fail before model loading, including `mtp` with a DFlash path, `dflash2`
without a checkpoint, and `none` with an enabled draft. `--mtp-chain` and MTP
adaptive-depth options configure native MTP only; DFlash derives its block size
from the draft checkpoint. The current Qwen3.8-27B checkpoint uses an anchor plus
up to seven proposed tokens.

Examples using a CUDA/FlashInfer build:

```sh
# Target only
./build-dflash-integration/qw3 serve --model models/Qwen3.8-27B-Q8_0.gguf \
  --kv-dtype fp8 --speculative-method none

# Native MTP
./build-dflash-integration/qw3 serve --model models/Qwen3.8-27B-Q8_0.gguf \
  --kv-dtype fp8 --speculative-method mtp --mtp-chain 4

# DFlash 2
./build-dflash-integration/qw3 serve --model models/Qwen3.8-27B-Q8_0.gguf \
  --kv-dtype fp8 --speculative-method dflash2 \
  --dflash-model models/Qwen3.8-27B-DFlash2-BF16
```

Use one command at a time, or separate ports. Add the existing KVMem configuration
to the chosen command; selecting a draft does not change KVMem budgets or enable
KVMem automatically. The draft must match the target model. The imported tested
configuration is Qwen3.8-27B Q8_0 target weights, FP8 target KV and BF16 DFlash
weights; this does not extend target-weight or KV-format support to BF16.

For C++ callers, set `EngineOptions::speculative_method` to `none`, `mtp` or
`dflash2`, and set `dflash_model_path` for DFlash. The default `auto` preserves
the existing `native_mtp_*` options. Explicit selection takes precedence over
legacy MTP environment activation. Rebuild consumers after the EngineOptions
and DeviceBackend interface changes.

## KVMem and concurrency

The import includes DFlash feature capture, cache/checkpoint rollback, stochastic
acceptance, KVMem replay and archive state, and the experimental continuous
controller. It also preserves the main tree's CacheBlend source and tests.
Existing experimental optimization switches retain their defaults.

DFlash continuous batching remains opt-in through `--continuous-batching` and
the existing `QW3_CONTINUOUS_DFLASH=1` switch. Full KVMem control uses
`QW3_CONTINUOUS_KVMEM_CONTROLLER=1`; sampling batches additionally use
`QW3_CONTINUOUS_BATCHING_SAMPLING=1`. Advanced HTTP session reuse and bounded
repository concurrency have further gates. These switches alone are not a
complete deployment configuration: the shared GPU pool and CPU/NVMe resources
must fit every active request. Unsupported combinations may use the serialized
route, or fail where a fallback would violate resource reservations.

The import does not promote joint prefill, MMA validation experiments or bounded
HTTP/GC lifecycle to qualified defaults. Historical measurements, including
negative performance and numerical results, remain in the `dflash2_*` reports.
Original benchmark artifacts referenced by those reports remain in the DFlash
development worktree; model files and build artifacts are not copied as source.

## Integration provenance and checks

The source was snapshotted from `/home/chaidi/qw3-dflash2`, HEAD `7ffa50c`,
including its uncommitted concurrency changes on 2026-09-29. It was merged with
the existing `/home/chaidi/qw3` work rather than replacing that worktree.
The original destination files, source snapshot, hashes and merge audit are in
`/tmp/qw3-dflash-integration/` on the integration host.

Independent build directories are `build-dflash-integration-host` (CPU tests)
and `build-dflash-integration` (CUDA 13, SM120a, FlashInfer). Existing binaries
and running services are not replaced.

Integration validation on 2026-09-29:

- CPU Release build and all 44 CTest checks passed, including the method
  selector, 18 invalid public CLI configurations, sampling, KVMem storage,
  repository lifecycle and CacheBlend policy checks.
- The continuous-batching benchmark helper's two Python tests passed.
- The CUDA 13 / SM120a / FlashInfer executable compiled and linked, together
  with DFlash operations, forward and backend probes, GPU sampling and
  continuous-sampling probes, and the CacheBlend row-mixing test executable.
  The CUDA-linked CLI also passed the 18 invalid-configuration checks and
  its help/option checks without running inference.
- The local `Qwen3.8-27B-DFlash2-BF16` checkpoint passed the model structure and
  tensor audit: 81 tensors, 5 draft layers, hidden size 5120, block size 8.
  The Q8_0 target's native plan reports supported MTP weights (15 tensors).
- GPU inference, acceptance/parity, KVMem replay and throughput were not rerun
  during this integration because another experiment occupied the GPU. The
  historical DFlash qualification reports describe the source worktree's
  results; they are not a new qualification of the merged executable.
