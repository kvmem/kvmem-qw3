# DeepSWE requestplan10 with official Pier + mini-swe-agent

This experiment re-runs the exact ten tasks from the historical KVMem 8/10
batch through DeepSWE's official execution path:

- Pier creates the official task container and verifier environment.
- Pier installs the pinned mini-swe-agent release and uses its bundled
  `mini.yaml` prompt, native bash tool schema, step loop, and completion marker.
- Each task uses one attempt for a direct comparison with the historical 8/10
  one-attempt batch. A separate four-attempt run is required for pass@4.
- QW3 is restarted for every task because the current serving implementation is
  intentionally single-trajectory. Pier concurrency is therefore one.
- QW3 permits a 3M-token logical trajectory backed by a 110 GiB CPU tier while
  retaining the 128K active KVMem selection budget and 64K generation reserve.
- A task is resumably skipped after a numeric official verifier reward is
  captured. A normal completion or a verifier-scored `AgentTimeoutError` is a
  valid one-attempt model outcome; reward zero is recorded and neither case is
  automatically retried. Other post-agent exceptions are recorded as task
  failures, but no longer prevent later tasks in the queue from running.
- A transient official container-build/network failure that occurs before
  MiniSweAgent starts is retried up to ten infrastructure attempts. Each
  attempt has a separate artifact directory. Failures after the agent starts,
  and every numeric verifier result including zero, are never auto-retried.

The provider is explicitly `litellm` so mini-swe-agent uses Chat Completions,
which is the native QW3 API. The request includes
`x-qw3-harness: mini-swe-agent`, allowing QW3 to use explicit harness semantics
instead of guessing from prompt text.

Pier's policy proxy intentionally permits only destination ports 80 and 443.
QW3 remains an unprivileged host process on port 8000; a task-base-image
container exposes a host-local TCP relay on port 80. The agent reaches it as
`172.17.0.1.nip.io`, so Pier keeps both domain allowlisting and network
isolation enabled. The relay contains no HTTP or model logic.

Pier's upstream MiniSweAgent adapter is inherited by a local reliability
subclass. Installation of the pinned `uv 0.7.13` uses curl HTTP/1.1 with
bounded retries, avoiding reproducible HTTP/2 truncation from GitHub on this
host. For slower local execution, shell commands receive 1800 seconds, model
requests receive 21600 seconds, and the consecutive format-error guard is
raised from 3 to 10. Step, cost, and MiniSweAgent wall-clock limits are disabled.
The MiniSweAgent version remains 2.4.6, and its prompt, tools, execution loop,
trajectory conversion, network policy, and verifier semantics are unchanged.
The adapter source hash and policy are stored in the run manifest.

## Locked components

See `versions.lock.json`. The runner validates the Pier and mini-swe-agent
versions and hashes both the DeepSWE task manifest and the authoritative task
selection before starting.

## Run

```bash
./benchmark/mini_swe_deepswe/run_requestplan10.py \
  --run-name requestplan10_pier_mini_20260823
```

For a one-task infrastructure smoke test:

```bash
./benchmark/mini_swe_deepswe/run_requestplan10.py \
  --run-name requestplan10_pier_mini_20260823 \
  --only expr-try-catch-errors
```

Re-running the same command resumes from official verifier-complete tasks.

For independent repeated rollouts, keep all other options fixed and vary the
recorded QW3 sampling seed.  For example, after the original seed-73 rollout:

```bash
./benchmark/mini_swe_deepswe/run_requestplan10.py \
  --run-name requestplan10_pier_mini_20260823_c \
  --seed 74 \
  --rerun <task-id>
```

Each attempt stores `rollout_seed` together with the exact QW3 command.  Using
the same seed for every fresh per-task service would make nominal pass@4
attempts unnecessarily correlated.

Local runs default to an operationally unbounded Pier agent deadline. Setup,
environment-build, and verifier timeouts are multiplied by four, while the
official collect-command limits remain intact. Use `--official-agent-timeout`
to restore the task's 90-minute agent limit. Specific completed tasks can be
rerun into new, separately retained infrastructure-attempt directories:

```bash
./benchmark/mini_swe_deepswe/run_requestplan10.py \
  --run-name requestplan10_pier_mini_20260823 \
  --rerun wasmi-trap-coredumps \
  --rerun opa-template-string-reconstruction
```

## Ordinary QW3 256K compaction baseline

The pinned mini-swe-agent 2.4.6 does **not** implement automatic context
compaction: its default agent retains and resends the complete `messages` list
until the provider rejects it. Consequently, changing only QW3's `--ctx` to
256K is not a valid compaction baseline.

The runner provides a separate `dense-compaction` mode for the matched
ordinary-QW3 comparison:

```bash
./benchmark/mini_swe_deepswe/run_requestplan10.py \
  --run-name requestplan10_pier_mini_dense256_compaction_20260823 \
  --mode dense-compaction
```

This mode keeps the same model, FP8 KV cache, sampling parameters, thinking
budget, MTP=4, official Pier task, MiniSWE prompt, bash tool, and verifier. It
disables KVMem and starts QW3 with a 262,144-token dense context. Before the
estimated next prompt reaches 220,000 tokens, the same QW3 model receives a
private, non-thinking compaction request capped at 16,384 output tokens. The
agent then retains the original system prompt and task verbatim, replaces the
intermediate transcript with that structured continuation record, and resumes
the ordinary MiniSWE loop. A provider context-window rejection is also caught
as a one-time emergency compaction path.

Every compaction is written into the MiniSWE trajectory under
`info.compaction.records`, including the trigger estimate, old/new message
counts, summary size, token usage, finish reason, and wall time. The console
also emits a `[qw3-compaction]` line, so the dashboard and post-processing can
verify that compaction actually occurred rather than inferring it from prompt
length.
