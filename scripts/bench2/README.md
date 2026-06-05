# bench2 — clean benchmark layer

The qw3 binary's only job is to run an OpenAI-compatible server (`qw3 serve`).
This package does ALL benchmarking from Python over HTTP. It never launches a
server, never reloads a model, and never parses stderr — timing is pure
wall-clock against a server URL you started yourself.

## 1. Start one or more servers

```bash
# fp16 (default KV)
./build/qw3 serve --model models/Qwen3.6-27B-Q8_0.gguf \
    --backend qwen-native --native-heavy --native-kernels cuda \
    --native-linear-backend auto -c 131072 --port 8080

# fp8 KV (opt-in), separate process for side-by-side comparison
./build/qw3 serve --model models/Qwen3.6-27B-Q8_0.gguf \
    --backend qwen-native --native-heavy --native-kernels cuda \
    --native-linear-backend auto --kv-dtype fp8 -c 131072 --port 8082
```

## 2. Benchmark them

```bash
# single server
python3 scripts/bench2/run.py \
    --endpoint qw3=http://127.0.0.1:8080/v1 \
    --prompt-tokens "4096 16384 65536" --n-decode 128

# fp8 vs fp16 side-by-side
python3 scripts/bench2/run.py \
    --endpoint fp16=http://127.0.0.1:8080/v1 \
    --endpoint fp8=http://127.0.0.1:8082/v1 \
    --prompt-tokens "65536 131072 256000" --n-decode 128 \
    --out /tmp/fp8_vs_fp16.json
```

The report prints prefill tok/s, decode tok/s, TTFT, and peak VRAM with one
column per endpoint (and an fp8/fp16-style ratio column when exactly two
endpoints are given).

## Timing methods

- `--method stream` (default): one streamed `/v1/chat/completions` request.
  The first content SSE chunk lands after prefill, so TTFT == prefill latency;
  the gaps between later chunks give decode tok/s.
- `--method two_point`: two non-streamed `/v1/completions` (max_tokens=1 and
  max_tokens=M). `prefill_s ≈ t(1)`, `decode_tok_s ≈ (M-1)/(t(M)-t(1))`. Use
  this if a server's completions endpoint does not stream.

All requests are greedy (temperature=0, seed=0).

## Files

- `client.py` — HTTP timing client (`measure_stream`, `measure_two_point`).
- `util.py` — `make_prompt` (token-calibrated) + `VramPoller` (nvidia-smi).
- `driver.py` — walks the (endpoint × prompt-length) grid, medians N trials.
- `report.py` — side-by-side text/markdown report.
- `run.py` — CLI entrypoint.
