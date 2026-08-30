# QW3 v1 Architecture

Native execution follows the `ds4-main` shape: the model loader and executor are
owned by QW3, while GPU weights, activations, scratch buffers, and command
lifetime are owned by the CUDA device layer. QW3 exposes one generation
runtime: its native CUDA implementation. llama.cpp remains outside this runtime
and is used only as an optional external benchmark baseline.

## v1 Scope

- Load and inspect GGUF metadata with a native lightweight GGUF reader.
- Load Qwen GGUFs with QW3's own mmap loader.
- Bind Qwen3.6/Qwen35 tensors into a native model plan.
- Keep external comparison tooling outside the runtime boundary; the
  benchmark script invokes a separately installed `llama-completion` process.
- Keep host-side model inspection and unit tests available in builds without
  CUDA; these builds do not provide generation.
- Keep CLI and prompt rendering independent from the underlying CUDA kernel
  implementation.

## Runtime Boundary

The native runtime owns GGUF/HF model loading, tokenizer state, Qwen execution
planning, and dispatch into the device-resident CUDA layer. Runtime selection is
not exposed: generation always follows this native CUDA path.

The native executor no longer uses CPU kernels as its correctness path. It talks
to `include/qw3/device_backend.hpp`, whose tensors and weights are backend-owned
device handles. CUDA kernels can therefore be replaced with CUTLASS,
llama.cpp-derived kernel code, or custom fused kernels without changing GGUF
loading or the Qwen execution plan.

## Running

Build the CUDA runtime:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real
cmake --build build -j
```

Build and run host unit tests without CUDA:

```sh
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

The host suite is not a runtime smoke. Before a release is marked tested, a
clean GPU gate must run the CUDA binary with a pinned real model and record the
model digest, build configuration, GPU, and result.

Run the CUDA component tests:

```sh
ctest --test-dir build --output-on-failure
```

Inspect GGUF:

```sh
./build/qw3-inspect /path/to/qwen.gguf
```

Generate with qw3:

```sh
./build/qw3 \
  --model /path/to/qwen.gguf \
  -p "写一个 CUDA matmul kernel 的优化清单" \
  -n 256 -c 32768
```

Use `--raw` to bypass the built-in Qwen chat formatting and send text exactly
as provided.

Run the optional external llama.cpp benchmark:

```sh
python3 scripts/long_prompt_sweep.py \
  --qw3 ./build/qw3 \
  --model /path/to/qwen.gguf \
  --llama /path/to/llama-completion \
  --prompt-tokens "512 1024 2048 4096" \
  --trials 3 -n 64
```

The benchmark launches qw3 and `llama-completion` independently; llama.cpp does
not enter the QW3 runtime.
