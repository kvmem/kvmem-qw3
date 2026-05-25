# QW3 v1 Architecture

This version keeps llama.cpp as a baseline/reference path and moves native
execution toward the `ds4-main` shape: the model loader and executor are owned by
QW3, while GPU weights, activations, scratch buffers, and command lifetime are
owned by a device backend.

## v1 Scope

- Load and inspect GGUF metadata with a native lightweight GGUF reader.
- Load Qwen GGUFs with QW3's own mmap loader.
- Bind Qwen3.6/Qwen35 tensors into a native model plan.
- Keep the llama.cpp executable backend only as a baseline/accuracy reference.
- Provide a mock backend for CI/build tests that do not have model weights or
  llama.cpp installed.
- Keep CLI, prompt rendering, and backend interfaces independent from the
  underlying kernel implementation.

## Backend Boundary

`src/backend.hpp` is the replacement point for later work. Current backends:

- `mock`: no model execution, used by smoke tests.
- `qwen-native`: owns GGUF mmap loading, tensor binding, Qwen execution
  planning, and dispatch into the device-resident CUDA backend. This is where
  kernel iteration should happen.
- `llama-cli`: invokes an existing `llama-completion` binary with the selected
  GGUF. This is a baseline/reference path, not the optimization path.

Backends implement the same `Backend` interface:

```cpp
class Backend {
public:
    virtual void load(const EngineOptions &options) = 0;
    virtual std::string generate(const std::string &prompt,
                                 const GenerationOptions &options,
                                 const TokenCallback &on_text) = 0;
};
```

The native executor no longer uses CPU kernels as its correctness path. It talks
to `include/qw3/device_backend.hpp`, whose tensors and weights are backend-owned
device handles. CUDA kernels can therefore be replaced with CUTLASS, llama.cpp
kernel code, or custom fused kernels without changing GGUF loading or the Qwen
execution plan.

## Running

Build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Smoke test:

```sh
ctest --test-dir build --output-on-failure
```

Inspect GGUF:

```sh
./build/qw3-inspect /path/to/qwen.gguf
```

Generate with llama.cpp:

```sh
./build/qw3 \
  --llama-completion /path/to/llama-completion \
  --model /path/to/qwen.gguf \
  -p "写一个 CUDA matmul kernel 的优化清单" \
  -n 256 -c 32768 -ngl -1
```

Use `--raw` to bypass the built-in Qwen chat formatting and send text exactly
as provided.
