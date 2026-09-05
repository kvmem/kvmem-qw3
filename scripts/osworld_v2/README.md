# QW3 × OSWorld-V2 local evaluation

This integration pins OSWorld to `v2026.08.08` and uses the local Docker/KVM
provider with QW3's OpenAI-compatible multimodal Chat Completions endpoint.

## Provisioning choices

- OSWorld code/tasks/assets: `osworld-v2-2026.08.08`
- Docker/QEMU image: the manifest-pinned `v2026.06.24` artifact with SHA-256
  `eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa`
- Docker runtime: `happysixd/osworld-docker` pinned to digest
  `sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9`
- Hosted benchmark websites: `site.hku.icu`
- Model: local `Qwen3.8-27B-Q8_0.gguf` with the native CUDA BF16 vision tower
- Agent adapter: OSWorld's bundled `Qwen35VLAgent`, using relative coordinates
  and four screenshots of action history
- Initial test: one environment, screenshot observation, pyautogui action

All Hugging Face downloads use `https://hf-mirror.com` and explicitly remove
inherited HTTP, HTTPS, and ALL proxy variables.

Exact source revisions, artifact digests, and byte sizes are recorded in
[`PROVENANCE.md`](PROVENANCE.md).
The latest local smoke-test outcome and the current full-run go/no-go decision
are recorded in [`VALIDATION.md`](VALIDATION.md).

## Prepare

Accept access to both gated Hugging Face datasets in a browser, then log in
locally without placing the token in source files or chat:

```bash
cd /home/chaidi/qw3/third_party/OSWorld-V2
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/hf auth login
```

Then run:

```bash
/home/chaidi/qw3/scripts/osworld_v2/download_pinned_assets.sh
/home/chaidi/qw3/scripts/osworld_v2/verify_setup.sh
```

The download script also audits all 108 task classes and writes the ignored
dependency report to `cache/qw3_task_audit.json`. It is used to choose a smoke
task that does not need GitLab, account credentials, a proxy, a user simulator,
or an LLM evaluator. The task downloader additionally verifies the pinned
official manifest itself and the SHA-256 digest of every task class.
The task path uses authenticated Git smart-HTTP against hf-mirror at a fixed
dataset commit. The mirror's ordinary `resolve/raw` text endpoints can alter
or truncate some Python responses, while Git transfers the canonical blobs;
both the temporary checkout and installed task set are hash-verified.
The asset path similarly repairs canonical non-LFS text blobs through Git,
then derives official binary hashes from the pinned Git LFS pointers and
verifies every one of the 1061 downloaded assets.

## Run

`run_qw3_smoke.sh` refuses to start while the current DeepSWE main process,
trial worker, or run-owner container is active. Once the gated task classes are
available, `evaluation_examples/test_qw3_smoke.json` selects audited task 052:
a deterministic TravelHub visible-UI booking flow with no account, GitLab,
proxy, user-simulator, or model-evaluator dependency.

```bash
/home/chaidi/qw3/scripts/osworld_v2/run_qw3_smoke.sh
```

The smoke deliberately uses a short 15-step diagnostic budget; it is not the
formal long-horizon profile and can be shortened with
`QW3_OSWORLD_MAX_STEPS=<n>`. Image requests intentionally use serialized
serving because QW3 does not support continuous batching for multimodal input.
Set `QW3_OSWORLD_ENABLE_VNC=1` to keep the VM's noVNC service available and
`QW3_OSWORLD_ENABLE_RECORDING=1` to save a recording when the selected task
allows those facilities. For the local Docker provider, read the dynamically
assigned host VNC port from `runner.log`.

## Trace and screenshot-compaction A/B

The controlled A/B runner repeats the same task with the same 100-step text and
action history. The `compact` condition keeps at most four live screenshots and
folds two old screenshots at each boundary; `no-compact` keeps up to 64. QW3
KVMem is disabled in both conditions, so this experiment isolates the GUI
adapter's screenshot-history folding rather than KVMem retrieval.

```bash
QW3_OSWORLD_EXPERIMENT_ID=<id> QW3_OSWORLD_MAX_STEPS=8 \
  /home/chaidi/qw3/scripts/osworld_v2/run_compaction_ab.sh
```

The runner removes uppercase and lowercase HTTP, HTTPS, and ALL proxy variables
and keeps `HF_ENDPOINT=https://hf-mirror.com`. Each condition records:

- exact processed model-input screenshots and sanitized full message payloads;
- one structured agent trace and one end-to-end timing record per model turn;
- every executed action, its dispatch/observation latency, and post-action image;
- QW3 request/accounting logs with preprocess, queue, engine, response, TTFT,
  native prefill, and native decode timings;
- guest memory, CPU, network, pressure, and top-process samples before, after
  every action, and before evaluation;
- source/runtime manifests, runner logs, evaluator timing, result, CSV/JSON
  analysis, Markdown report, and SVG latency chart.

Audit and seal a condition with a SHA-256 manifest using:

```bash
/home/chaidi/qw3/third_party/OSWorld-V2/.venv/bin/python \
  /home/chaidi/qw3/scripts/osworld_v2/verify_trace_bundle.py \
  --run-dir <condition-dir> --expected-steps 8 \
  --expected-active-images 1,2,3,4,3,4,3,4 \
  --expected-compaction-steps 5,7
```

The measured 2026-09-03 experiment and its limitations are documented in
[`VALIDATION.md`](VALIDATION.md).

After smoke validation, the complete 108-task entry point is:

```bash
/home/chaidi/qw3/scripts/osworld_v2/run_qw3_full.sh
```

Formal QW3 runs default to the `osworld-v2-public-pyautogui-200` profile:
200 model turns per phase/task, matching the public-evaluation example for an
OpenAI-style pyautogui agent. OSWorld-V2 does not publish one step budget for
every action space—the Claude computer-use example uses 500—so the profile is
recorded explicitly in every `run_manifest.json`. Changing
`QW3_OSWORLD_MAX_STEPS` creates an exploratory, non-profile run unless its
profile label is also changed deliberately.

Without residential-proxy credentials, use the deterministic 100-task subset
that excludes the eight official `proxy=true` tasks (`036`, `037`, `050`,
`055`, `056`, `062`, `075`, and `098`):

```bash
/home/chaidi/qw3/scripts/osworld_v2/run_qw3_no_proxy.sh
```

The full runner requires `GITLAB_URL` and `GITLAB_PRIVATE_TOKEN` in the local
mode-0600 export file. The official self-hosting source is pinned locally at
`third_party/OSWorld-GitLab`, and all three service images are pinned by digest.
After DeepSWE exits, `start_local_gitlab.sh` can create an isolated bridge at
`172.30.0.1`, bind GitLab only to that private interface, generate a local
mode-0600 token file, and register the bundled Runner. The repository's proxy
config still contains placeholders; the eight proxy-marked tasks require
compatible residential-proxy credentials for a valid complete 108-task run.

```bash
/home/chaidi/qw3/scripts/osworld_v2/start_local_gitlab.sh
```

`stop_local_gitlab.sh` stops the services while retaining their volumes and
generated credentials.

Model-based evaluators default to the same local QW3 endpoint so the suite can
run without a paid API. Those scores are useful for local engineering but are
not directly comparable with scores produced by a reference evaluator. Set the
`OSWORLD_EVAL_MODEL_*` variables to use a reference evaluator instead.
