"""Clean benchmark layer for OpenAI-compatible servers.

The qw3 binary's only job is to run `qw3 serve` (an OpenAI-compatible HTTP
server). This package does ALL benchmarking from Python against an
already-running server URL: it never launches a server, never parses stderr,
and never reloads a model. Timing is pure wall-clock over HTTP.

Launch a server yourself, e.g.:
    ./build/qw3 serve --model models/Qwen3.6-27B-Q8_0.gguf -c 8192 --port 8080
    ./build/qw3 serve ... --kv-dtype fp8 --port 8082
    ./build/qw3 serve ... --native-mtp-speculate --native-mtp-chain 3 --port 8083

then point the bench at one or more of them:
    python3 scripts/bench2/run.py \
        --endpoint fp16=http://127.0.0.1:8080/v1 \
        --endpoint fp8=http://127.0.0.1:8082/v1 \
        --prompt-tokens "4096 16384 65536" --n-decode 128 --out /tmp/bench2.json
"""
