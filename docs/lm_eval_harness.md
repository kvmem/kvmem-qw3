# lm-evaluation-harness

Benchmark evaluation should use EleutherAI `lm-evaluation-harness` directly.
This repository does not keep a custom evaluation runner.

## Install

```sh
source .venv/bin/activate
python -m pip install "lm_eval[api,vllm]"
```

## Qwen3.6-27B GSM8K (Official-style, vLLM + Thinking)

This matches Qwen math-eval conventions: thinking mode, chat template,
`\boxed{}` prompt, and `\boxed{}` answer extraction. Config lives in
`benchmark/gsm8k_qwen36_official.yaml`; the custom task is
`benchmark/tasks/gsm8k_qwen36_official.yaml`.

Full run (1319 test samples, ~1–2 hours):

```sh
source .venv/bin/activate
lm_eval run --config benchmark/gsm8k_qwen36_official.yaml
```

Smoke test (5 samples):

```sh
source .venv/bin/activate
lm_eval run --config benchmark/gsm8k_qwen36_official.yaml \
  --limit 5 \
  --output_path benchmark/lm_eval_qwen36_27b_vllm_official_smoke
```

Key settings vs the default `gsm8k` task:

| Setting | Default `gsm8k` | Official-style |
|---------|-----------------|----------------|
| Backend | HF / local server | vLLM |
| Thinking | off | `enable_thinking: true` |
| Chat template | off | `apply_chat_template: true` |
| Prompt | `Question: … Answer:` | `\boxed{}` instruction |
| `max_model_len` | 8192 | 16384 |
| `max_gen_toks` | 2048 | 8192 |
| Primary metric | `strict-match` (`####`) | `boxed-match` (`\boxed{}`) |

Other variants in `benchmark/`:

- `gsm8k_qwen36_thinking.yaml` — thinking mode with the stock `gsm8k` task
  (no `\boxed{}` prompt; uses `####` extraction)

## Full Evaluation (Local GGUF Server)

For the local OpenAI-compatible server at `http://127.0.0.1:8080`, use the
concrete completions endpoint `http://127.0.0.1:8080/v1/completions`:

```sh
source .venv/bin/activate
mkdir -p benchmark/lm_eval_qwen36_27b

lm_eval run \
  --model local-completions \
  --model_args model=Qwen3.6-27B-Q8_0.gguf,base_url=http://127.0.0.1:8080/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,timeout=1200 \
  --tasks gsm8k,mmlu \
  --batch_size 16 \
  --output_path benchmark/lm_eval_qwen36_27b \
  --log_samples
```

## Limited Check

Use `--limit` only when you intentionally want a short debug run:

```sh
source .venv/bin/activate
mkdir -p benchmark/lm_eval_qwen36_27b_limit20

lm_eval run \
  --model local-completions \
  --model_args model=Qwen3.6-27B-Q8_0.gguf,base_url=http://127.0.0.1:8080/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,timeout=1200 \
  --tasks gsm8k,mmlu \
  --batch_size 16 \
  --limit 20 \
  --output_path benchmark/lm_eval_qwen36_27b_limit20 \
  --log_samples
```

## Chat Endpoint Variant

Only use this if the server's `/v1/chat/completions` route is known to be
working correctly for Qwen:

```sh
source .venv/bin/activate
mkdir -p benchmark/lm_eval_qwen36_27b_chat

lm_eval run \
  --model local-chat-completions \
  --model_args model=Qwen3.6-27B-Q8_0.gguf,base_url=http://127.0.0.1:8080/v1/chat/completions,num_concurrent=1,max_retries=3,timeout=1200 \
  --tasks gsm8k,mmlu \
  --batch_size 16 \
  --output_path benchmark/lm_eval_qwen36_27b_chat \
  --log_samples
```
