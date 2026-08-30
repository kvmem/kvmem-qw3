# Open-Source Release Baseline

> Status: release preparation. This document is intentionally conservative and
> must be updated from a clean, tagged source snapshot before each release.

This document defines what the project may claim in public documentation. A
code path existing, or having been used in an internal experiment, is not by
itself sufficient for public support.

## Status vocabulary

- **Tested**: reproduced from the clean source snapshot below with the exact
  command and result recorded here.
- **Expected**: the implementation accepts this path, but it has not passed the
  open-source release gate.
- **Experimental**: an implemented research path whose interfaces,
  constraints, or results may change.
- **Unsupported**: rejected by the current CLI, build, or runtime, or no native
  implementation exists.

`Tested` is deliberately narrower than "used during development." Internal
benchmark notes do not make a configuration release-tested.

## Clean host-test baseline

| Snapshot | Environment | Configure/build | Tests | Runtime coverage | Status |
|---|---|---|---|---|---|
| `e0e09e3` plus the current uncommitted open-source preparation changes (fresh build, 2026-08-23) | Linux x86-64, CMake 3.22.1, GCC 11.4.0, CUDA disabled | Release build passed with `-j2` | 12/12 current host CTest targets passed | None; CUDA and a real model are required | Tested |

Commands:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

This verifies the host build and host-side unit tests. It does **not** verify
the QW3 runtime, which requires a CUDA build, a supported NVIDIA GPU, and a real
model.

Before any runtime row can move to **Tested**, the release gate must pin the
model URL/revision and SHA-256, build from the release snapshot, and record a
successful real-model smoke on the claimed GPU configuration.

Because the worktree has not been committed, the same check must be rerun from
the eventual release commit before the status can remain Tested.

## Model, weight, and hardware matrix

| Model/source | Native weight path | Hardware/build | Status | Basis and limitation |
|---|---|---|---|---|
| Qwen3.6 27B, GGUF v3 | Q8_0 weights with F32 tensors | NVIDIA CUDA; development records use RTX PRO 6000 Blackwell/SM120 | Expected | Extensively exercised internally, but no release-pinned model URL, revision, SHA-256, or clean GPU gate exists yet. |
| Qwen3.8 27B, GGUF v3 | Q8_0 weights with F32 tensors | NVIDIA CUDA | Expected | The implementation path exists, but there is no checked-in release-grade successful-run artifact. |
| Other `qwen35` or legacy `qwen3` GGUF models and sizes | Q8_0 and F32 tensors only | NVIDIA CUDA | Experimental | Metadata architectures are accepted, but model and shape coverage is not release-validated. |
| HF `qwen3_5_text` directory using `compressed-tensors` | Mixed NVFP4 E2M1, FP8 E4M3, BF16, and F32 | NVIDIA SM120a with FlashInfer and CUTLASS when NVFP4 is present | Experimental | Implemented, but the checkpoint and FlashInfer/CUTLASS stack are not pinned. Checkpoint parity is only registered when `QW3_NVFP4_TEST_MODEL` is supplied. |
| GGUF Q4, IQ, or other native tensor formats | — | Any | Unsupported | The native GGUF model source maps only GGML type 8 (Q8_0) and type 0 (F32). |
| Native generation without CUDA | — | Host-only build | Unsupported | Host builds support unit tests and model inspection; generation requires CUDA. |
| CUDA architectures below SM80 for native Q8 prefill | Q8_0 | NVIDIA below Compute Capability 8.0 | Unsupported | The INT8 MMA build path requires SM80, which is also the default CUDA architecture. |
| Ampere, Ada, or Hopper Q8_0 | Q8_0 and F32 tensors | NVIDIA Compute Capability 8.0+ | Expected | Intended by the build target, but no open-source release gate covers these GPUs yet. |
| Non-NVIDIA accelerators | — | AMD, Intel, Apple, or other accelerators | Unsupported | No native device backend exists. |

llama.cpp is not part of the QW3 runtime or native support promise.
The optional `scripts/long_prompt_sweep.py` development benchmark launches a
separately installed `llama-completion` executable directly.

## Runtime combination matrix

| Combination | Status | Current boundary |
|---|---|---|
| Plain QW3 CLI or serialized HTTP serving | Expected | Native CUDA path; it must pass a pinned clean GPU/model gate before becoming Tested. |
| Plain plus MTP | Experimental | Native MTP speculative decode is exposed as an experimental path. |
| Continuous batching without MTP | Experimental | Requires paged KV; broader sampling and concurrency release coverage is missing. |
| Continuous batching plus MTP | Experimental | Uses batched draft execution and paged MTP prefix state. |
| KVMem, serialized, without MTP | Experimental | Research path; `--kvmem` is off by default. |
| KVMem plus MTP | Experimental | Implemented and internally exercised, but not a release-stable contract. |
| KVMem plus continuous batching | Experimental | Per-request KVMem executors exist; prefix-cache and query-replay behavior differs from the serialized route. |
| KVMem plus continuous batching plus MTP | Experimental | Only the window-aware ragged verifier is supported; the opt-in layered verifier is rejected. |
| Frozen archive build, query, or serve | Experimental | FP8-only; forces KVMem, immutable source K, raw-K NVMe authority, and defaults MTP chain to 4. Serving is serialized. |
| Frozen archive plus continuous batching | Unsupported | The CLI rejects this combination. |
| Frozen archive with non-FP8 KV | Unsupported | The CLI and runtime reject it. |

## KVMem retrieval and index placement

The table below assumes:

```text
--kvmem --kvmem-method retrieval --kvmem-query-conditioned
```

All accepted entries remain Experimental until a clean, pinned GPU gate is
published.

| `--kvmem-retrieval-method` | GPU index | CPU index | Notes |
|---|---|---|---|
| `mean-k` | Experimental | Experimental | CPU placement streams bounded tiles and requires query-conditioned mode. |
| `per-token` | Experimental | Unsupported | Runtime explicitly rejects per-token retrieval with CPU index placement. |
| `sub-block-mean-k` | Experimental | Experimental | Both placements are implemented. |
| `key-direction-fixed4` | Experimental | Experimental | Uses the sub-block scorer with fixed four-direction prototypes. |
| `key-direction-adaptive` | Experimental | Experimental | Packed-GPU and streamed-CPU scoring paths exist; internal A/B evidence is not a public release gate. |

These five rows are the only public CLI scorer values. Names such as
`mean_attention`, `content_mean`, and the internal DeltaNet scorer are not
accepted public values. `--kvmem-method h2o` and `recency` do not use a
retrieval index, so index placement is not applicable to them.

## Open-source release blockers

The following items must be closed before a public Research Preview tag:

1. Publish exact model repository, revision, file hash, acquisition or
   conversion procedure, and model license for every advertised profile.
2. Pin one compatible FlashInfer/CUTLASS stack and record CUDA, compiler, and
   GPU architecture requirements. The existing local CUDA build mixes
   unpinned dependency sources and is not release evidence.
3. Run at least one clean GPU gate per advertised profile and publish the
   command, environment, and result artifact.
4. Replace or remove README references to missing scripts and obsolete KVMem
   scorer names.
5. Repair the benchmark source allow-list. Some tracked runners require
   ignored files, while several currently ignored lm-eval YAML/Python
   placeholders contain only zero or one byte and must not be published as
   working configurations.
6. Resolve provenance for the tokenizer logic described as matching
   llama.cpp and for MTP code comments that identify `qw3_ly` as their source.
   These have not been classified as third-party code without owner evidence.
7. Add automated link, snippet, secret, and private-path checks to the release
   gate. The current audit found developer-specific absolute paths in 166
   tracked files; stable user documentation must eliminate them, while
   historical research records need an explicit archive policy.
8. Complete a security and freshness review of vendored dependencies before
   exposing the HTTP server publicly, with particular attention to the
   network-facing cpp-httplib 0.15.3 header.

## Evidence sources

The current classifications are derived from the following implementation and
test boundaries:

- `CMakeLists.txt`: CUDA, SM80, NVFP4, optional dependencies, and test gates;
- `src/model_source.cpp`, `src/gguf.cpp`, and `src/qwen_config.cpp`: accepted
  model containers, architectures, and tensor types;
- `src/qw3_cli.cpp`: public option values and archive constraints;
- `src/qwen_native_backend.cpp`: CUDA requirement, KVMem placement checks,
  continuous batching interactions, and archive defaults;
- `tests/` and the registered CTest targets: host and CUDA component coverage;
- `scripts/kvmem_e2e_regression.py` and continuous-batching regression scripts:
  development-only end-to-end evidence.

This list identifies where evidence comes from; it does not turn every covered
path into a Tested public configuration.
