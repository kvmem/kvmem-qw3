# OpenHands SWE-bench Evaluation

This workflow runs headless OpenHands SWE-bench inference against an already
running `qw3 serve` endpoint, then scores the generated patches with the
official SWE-bench harness.

It is intended for KVMem-style experiments where OpenHands supplies the agent
workspace and optional context compaction, while `qw3` supplies the model
serving and KV/runtime instrumentation.

## Prerequisites

Start `qw3` before launching the benchmark:

```sh
./build/qw3 serve \
  --model /path/to/Qwen3.6-27B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx 65536 \
  --continuous-batching \
  --paged-kv
```

Install Docker for local SWE-bench workspaces and make sure `uv` is available.
The wrapper can clone OpenHands benchmarks on first use. It uses SSH by default
(`git@github.com:OpenHands/benchmarks.git`) so your GitHub SSH key must be
available:

```sh
python3 scripts/run_openhands_swebench.py --clone-openhands --skip-infer \
  --existing-output-json /path/to/existing/output.jsonl \
  --skip-evaluation
```

For a real rollout, either keep the default clone location
`third_party/openhands-benchmarks` or pass `--openhands-dir`.
Override the clone URL with `--openhands-repo` if needed.

## Smoke Run

Run one SWE-bench Lite instance with OpenHands' LLM summarizing condenser:

```sh
python3 scripts/run_openhands_swebench.py \
  --clone-openhands \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --workspace docker \
  --n-limit 1 \
  --num-workers 1 \
  --eval-workers 1 \
  --max-iterations 100 \
  --memory-policy condenser
```

The script assumes `qw3` is at `http://127.0.0.1:8080/v1`. When using Docker
workspaces on Linux, the OpenHands agent-server container must reach the host
service through the Docker bridge, so start `qw3` with `--host 0.0.0.0` and pass
`--base-url http://172.17.0.1:8080/v1`.

## Comparing Compaction

Use the same dataset selection and run two passes:

```sh
# Compact baseline: OpenHands LLMSummarizingCondenser
python3 scripts/run_openhands_swebench.py \
  --select benchmark/swebench_instances.txt \
  --memory-policy condenser \
  --condenser-max-size 120 \
  --condenser-keep-first 4

# No-compaction / long-context baseline
python3 scripts/run_openhands_swebench.py \
  --select benchmark/swebench_instances.txt \
  --memory-policy no_condenser
```

`condenser` maps to OpenHands' `--enable-condenser`; `no_condenser` maps to
`--disable-condenser`. The generated OpenHands event logs contain
`Condensation` events with `forgotten_event_ids`, which are the compacted events
to align with `qw3` prefill/KV traces.

## Docker Images

OpenHands Docker workspace needs SWE-bench agent-server images. The wrapper does
not build them by default because this can be expensive. If the images are not
present locally, build them explicitly:

```sh
python3 scripts/run_openhands_swebench.py \
  --clone-openhands \
  --build-images \
  --dataset princeton-nlp/SWE-bench_Lite \
  --n-limit 1 \
  --memory-policy condenser
```

For large evaluations, build images once from the OpenHands benchmarks clone and
reuse them across runs.

If DockerHub is unavailable from the host, point SWE-bench base images at an
internal mirror:

```sh
python3 scripts/run_openhands_swebench.py \
  --swebench-image-template "registry.example.com/swebench/sweb.eval.{arch}.{repo}_1776_{name}:latest" \
  --strip-dockerfile-syntax \
  --dataset princeton-nlp/SWE-bench_Lite \
  --n-limit 1
```

`--strip-dockerfile-syntax` removes OpenHands' `# syntax=docker/dockerfile:1.7`
line from the local checkout so buildx does not need to pull the Dockerfile
frontend from DockerHub. The Docker build still needs access to the selected
SWE-bench base image and any package registries used inside the image build.

If you cannot edit `/etc/docker/daemon.json`, pull through a mirror and re-tag
the image locally:

```sh
docker pull docker.m.daocloud.io/library/python:3.13-bookworm
docker tag docker.m.daocloud.io/library/python:3.13-bookworm python:3.13-bookworm

docker pull docker.1ms.run/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-25500:latest
docker tag docker.1ms.run/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-25500:latest \
  swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-25500:latest
```

Local OpenHands builds may produce tags with only the SDK short SHA while
SWE-bench inference expects a phased tag. Pass `--image-tag-prefix <sdk-short-sha>`
to make both sides use the same tag prefix.

## Outputs

Results are written under:

```text
benchmark/results/<git-sha>_<timestamp>/openhands_swebench/
```

Important files:

- `llm_config.json`: OpenHands LLM config pointing at `qw3`.
- `infer.log`: OpenHands rollout log.
- `eval.log`: conversion plus official SWE-bench evaluation log.
- `output.jsonl`: copied OpenHands rollout output when inference ran.
- `predictions.swebench.jsonl`: SWE-bench-format predictions.
- `*.report.json`: official SWE-bench report when scoring completed.
- `summary_openhands_swebench.md`: run manifest and artifact paths.

The OpenHands output/trajectory is the source for agent-level metadata
(`Condensation`, tool calls, patch text). The `qw3` serving logs or traces should
be joined by request id/time window to compute KV-level metrics such as repeated
prefill tokens, page residency, reload/splice/repair cost, and TTFT.

## Existing Rollouts

To only convert and score an existing OpenHands `output.jsonl`:

```sh
python3 scripts/run_openhands_swebench.py \
  --skip-infer \
  --existing-output-json /path/to/output.jsonl \
  --dataset princeton-nlp/SWE-bench_Lite \
  --run-id qw3_existing_rollout
```

To only convert to predictions without running tests:

```sh
python3 scripts/run_openhands_swebench.py \
  --skip-infer \
  --existing-output-json /path/to/output.jsonl \
  --skip-evaluation
```
