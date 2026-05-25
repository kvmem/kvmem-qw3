# QW3

QW3 is a first-pass local inference framework for Qwen GGUF models, shaped to
stay easy to optimize later. The `qwen-native` path owns GGUF loading, tensor
binding, model planning, and the future execution graph in this repository. The
llama.cpp executable backend is kept only as a baseline and correctness
reference.

The goal of this version is not to outperform llama.cpp. The goal is to create
the native engine boundary where individual kernels can be replaced with
llama.cpp/GGML, CUTLASS, or custom CUDA implementations without handing model
execution to an external runner.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA build:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DQW3_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

## Run With llama.cpp

Install or build llama.cpp separately, then point QW3 at the `llama-completion`
binary:

```sh
./build/qw3 \
  --llama-completion /path/to/llama-completion \
  --model /path/to/qwen.gguf \
  -p "解释一下 FlashAttention 的核心思想" \
  -n 256 \
  -c 32768 \
  -ngl -1
```

Inspect a GGUF without llama.cpp:

```sh
./build/qw3-inspect /path/to/qwen.gguf
```

Run the framework without a model:

```sh
./build/qw3 --backend mock -p hello
```

Build the native Qwen tensor plan:

```sh
./build/qw3 --model /path/to/qwen.gguf --native-plan
./build/qw3 --backend qwen-native --model /path/to/qwen.gguf -p smoke
```

Run native CUDA single-token forward:

```sh
./build-cuda/qw3 \
  --backend qwen-native \
  --model /path/to/qwen.gguf \
  --native-heavy \
  --native-kernels cuda \
  --native-linear-backend auto \
  --native-token-id 0 \
  -p smoke -n 1
```

`--native-linear-backend` options:

- `auto`: choose cuBLAS path for large linear ops, custom kernel fallback
- `cublas`: force cuBLAS path (fallback to custom on OOM/error)
- `custom`: force custom CUDA matvec kernel

## Benchmark

Use `scripts/benchmark_compare.sh` to compare:

- `qwen-native(cublas)`
- `qwen-native(auto)`
- `llama.cpp` (`llama-completion`)

Metrics:

- TTFT (single-token run wall time)
- decode tok/s
- decode latency (ms/token)

Example:

```sh
scripts/benchmark_compare.sh \
  --model /path/to/qwen.gguf \
  --llama-cli /path/to/llama-completion \
  --qw3-bin ./build-cuda/qw3 \
  --prompt "Write a concise CUDA optimization checklist." \
  --ctx 32768 \
  --ngl -1 \
  --batch 2048 \
  --decode-tokens 32 \
  --out-dir ./bench-out
```

The script writes `summary.txt` and raw logs under `--out-dir`.

## Optimization Path

The replacement points are `src/backend.hpp` for model execution backends and
`include/qw3/device_backend.hpp` for device-resident tensors, weights, command
lifetime, and kernels. `qwen-native` is the path for ongoing kernel work; the
llama.cpp executable backend is just a baseline/accuracy reference.

See [docs/architecture.md](docs/architecture.md) for the v1 boundary.
