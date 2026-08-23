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
- A task is resumably skipped only after a numeric official verifier reward was
  captured with no Pier trial exception. A reward of zero is a valid result and
  is not automatically retried.

The provider is explicitly `litellm` so mini-swe-agent uses Chat Completions,
which is the native QW3 API. The request includes
`x-qw3-harness: mini-swe-agent`, allowing QW3 to use explicit harness semantics
instead of guessing from prompt text.

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
