# Open-source RC0 release inventory

Status: review proposal; no files have been removed, moved, staged, or committed by
this inventory.

Audit date: 2026-08-24

Audited private snapshot: `e0e09e311f825befa40a4fd28b321dcf111157a2`
plus the current working-tree changes. The final public snapshot must be audited
again after those changes are committed.

## Purpose

This document defines the intended boundary of the first public source release.
It separates the product source and reproducible release checks from local model
assets, generated results, and historical research material.

The classifications are:

- **Keep**: belongs in the RC0 public source tree.
- **Keep after repair**: useful public material, but not publishable until the
  listed blockers are fixed.
- **Archive**: useful historical or research material that should be moved to a
  clearly labelled archive or a separate research repository. It is not part of
  the RC0 supported workflow.
- **Exclude**: generated, local, private-path-heavy, incomplete, or otherwise not
  suitable for the public source tree.
- **Add**: missing release-governance or reproducibility material that must be
  created.

## Audit snapshot

The current repository boundary is not safe to publish as-is:

| Item | Observed state |
|---|---:|
| Tracked paths | 407 |
| Tracked scripts | 250 |
| Tracked `src/` paths | 55 |
| Tracked docs | 55 |
| Tracked tests | 25 |
| Non-ignored untracked paths | 64 |
| Tracked paths containing developer-home or private data-root paths | 166 |
| Tracked Markdown files containing those private paths | 26 |
| Current working-tree changes | 27 tracked paths plus 64 untracked paths |
| Local non-release assets | more than 220 GiB |

The largest local categories are approximately 151 GiB under `models/`, 57 GiB
under `benchmark/`, 10 GiB under `.venv*`, 1.7 GiB under `results/`, and more
than 1 GiB of build directories. The ignored benchmark tree contains more than
one million files and includes directories that were not readable during this
audit.

Therefore:

1. Do not archive or copy the current workspace as a release artifact.
2. Do not run `git push --mirror`, `git push --all`, or publish local Codex refs.
3. Construct the release only from one audited Git commit tree.

## Recommended public history boundary

Use a new, sanitized public root commit for RC0.

The current branch history reachable from `HEAD` contains about 196 MiB of blob
history and no blob of 10 MiB or larger. However, all local refs together expose
about 1.4 GiB of blob records, including results, build products, session JSON,
SQLite databases, and logs under private Codex refs. A normal push of one
explicit branch would not include those refs, but a mirror push would.

Recommended construction:

1. Form one audited release-candidate commit in the private repository.
2. Export only that tree with `git archive <RC_SHA>`.
3. Scan and build the exported tree in a disposable clean directory.
4. Initialize a new public repository and use the audited tree as its first
   commit.
5. Preserve author attribution in `CONTRIBUTORS.md` and in third-party notices.

Keeping the existing single-branch history is possible only after a formal scan
of every object reachable from the exact public ref and confirmation of author
and code provenance. It must still be pushed as one explicit ref, never as a
mirror.

## Root files

### Keep

- `.gitignore`, after the boundary fixes below.
- `CMakeLists.txt`, after dependency and test-gate fixes.
- `README.md`, after it is rewritten as the public entry point.
- `LICENSE` (Apache-2.0).
- `THIRD_PARTY_NOTICES.md`.
- `LICENSES/llama.cpp.txt`.
- `LICENSES/nlohmann-json.txt`.
- `LICENSES/cpp-httplib.txt`.

### Exclude or use only as editing input

- `README_V0.md`: use its shorter structure as input to the final `README.md`,
  but do not publish two competing root READMEs. It currently calls unverified
  GPU profiles “recommended” and acknowledges that model artifacts are not yet
  published.
- Root-local JSONL, PDF, PNG, logs, and machine-specific experiment outputs,
  including files hidden only by `.git/info/exclude`.

### Add before the public release

- `CONTRIBUTING.md`.
- `SECURITY.md` with a private reporting contact.
- `SUPPORT.md` with the tested/experimental/unsupported matrix.
- `CODE_OF_CONDUCT.md`.
- `CONTRIBUTORS.md` if a sanitized root commit is used.
- A machine-readable release profile containing dependency, model, and hardware
  pins.

## Product source and headers

### Keep

- All current files under `include/qw3/`.
- All existing `src/` files referenced by the current CMake build graph.
- Conditional FlashInfer, NVFP4, FP8, and SM120 AOT adapters while their CMake
  options remain public.
- `src/kernel_legacy.cu`: despite its name, it is part of the unconditional CUDA
  source list.
- The internal `Backend` interface may remain as an implementation seam; there
  is no longer a public runtime backend selector.

The following already-deleted tracked files must be recorded as deletions in the
release commit, otherwise a clean clone will not match the audited working tree:

- `src/backend.cpp`.
- `src/llama_cli_backend.cpp`.
- `src/mock_backend.cpp`.
- `tests/smoke.cpp`.

### Keep after repair

- Pin and validate the FlashInfer/CUTLASS/CCCL source and version used by the
  public build. The current local build mixes a modified FlashInfer checkout
  with CUTLASS from a Python wheel and is not reproducible.
- Make CMake fail fast when the requested CUDA architecture or external include
  layout is incomplete.
- Record one exact supported CUDA/compiler/GPU profile rather than inferring a
  support promise from code paths alone.

### Provenance checks required

- Confirm the ownership and origin of the code described as “ported from
  `qw3_ly`” in `src/qwen_native_backend.cpp` and `src/qwen_executor.cpp`.
- Confirm whether the tokenizer logic described as matching llama.cpp is an
  independent implementation or a derivative requiring additional attribution.
- Confirm whether the FlashInfer adapters are API integration code only or
  contain copied/modified upstream implementation.
- Review `cpp-httplib` 0.15.3 for security/freshness before exposing the HTTP
  server as a supported public service.

## Vendored third-party source

### Keep

Only the two vendored headers actually present in the build tree:

- `third_party/json.hpp`: nlohmann/json 3.11.3, MIT, including its preserved
  nested attributions.
- `third_party/httplib.h`: cpp-httplib 0.15.3, MIT.

The five llama.cpp-derived CUDA translation units plus
`src/cuda_helpers.cuh` remain covered by `LICENSES/llama.cpp.txt` and the
llama.cpp entry in `THIRD_PARTY_NOTICES.md` because that code remains in the
CUDA build.

FlashInfer, CUTLASS, CUDA, OpenSSL, and external `llama-completion` tools are not
vendored source entries. They must instead be recorded in the pinned build
manifest and, for binary releases, in an SBOM/binary third-party notice.

## Tests

### Keep as pass/fail tests

Host tests:

- `tests/prompt_render_test.cpp` (currently untracked; must be added).
- `tests/kvmem_store_test.cpp`.
- `tests/kvmem_prefix_reuse_policy_test.cpp`.
- `tests/kvmem_refresh_policy_test.cpp`.
- `tests/kvmem_request_plan_test.cpp`.
- `tests/tool_call_stream_test.cpp`.
- `tests/anthropic_adapter_test.cpp`.
- `tests/harness_semantics_test.cpp`.
- `tests/tokenizer_pretokenizer_test.cpp`.
- `tests/pinned_kv_tier_test.cpp`.
- `tests/nvme_kv_tier_test.cpp`.
- `tests/kvmem_archive_test.cpp`.

CUDA component tests:

- `tests/mmq_parity.cu`.
- `tests/rope_remap_parity.cu`.
- `tests/rope_remap_batched_parity.cu`.
- `tests/kvmem_assembly.cu`.
- `tests/kvmem_immutable_k.cu`.
- `tests/kvmem_deltanet_score.cu`.
- `tests/kvmem_softmax_pages.cu`.
- `tests/kvmem_kmean_merge.cu`.
- `tests/mixed_linear_parity.cu` when its build requirements are enabled.

### Archive as diagnostics or benchmarks

- `tests/rope_remap_drift.cu`: intentionally diagnostic and built without being
  registered as a test.
- `tests/nvfp4_decode_bench.cu`: a performance benchmark, not a pass/fail test.
- `tests/vmed_bench.cu`: a standalone microbenchmark not present in CMake.
- `tests/test_memoryagentbench_baselines.py`: not registered, loads research
  modules dynamically, and contains a local absolute model path.

### Test-gate repairs

- CUDA tests that find no GPU currently print a message and return success, so a
  CUDA job can be falsely green. Use a real skip return code for developer runs,
  and make the release GPU job fail if any required CUDA test is skipped.
- Add CTest labels, timeouts, and GPU serialization/resource locks.
- Keep host unit tests as a fast logic gate, but do not describe them as runtime
  inference smoke tests.
- Runtime smoke must use a pinned real model on a supported NVIDIA GPU.

## Release and regression scripts

### Keep after repair: minimum smoke dependency closure

- `scripts/_benchmark_common.sh`.
- `scripts/run_benchmark_smoke.sh`.
- `scripts/paged_kv_regression.py`.
- `scripts/continuous_batching_regression.py`.
- `scripts/continuous_prefill_interleave_regression.py`.
- `scripts/mtp_continuous_regression.py`.
- `scripts/kvmem_e2e_regression.py`.
- `scripts/continuous_batching_benchmark.py`.
- `scripts/mtp_throughput_benchmark.py`.
- `benchmark/manifest.env.example` (currently untracked; must be added if this
  harness remains the release entry point).

Repairs required:

- `run_benchmark_smoke.sh` hard-codes `build/` for CTest and can silently skip
  it. Derive the build directory from the selected executable or require an
  explicit `BUILD_DIR`.
- Add a strict GPU/device preflight and fail on skipped required CUDA tests.
- Split the approximately 30-minute benchmark smoke from a short, deterministic
  release smoke. The short gate should load one pinned Q8 model, perform a
  greedy CLI generation, perform a serialized HTTP generation, validate output
  and token accounting, and save provenance artifacts.
- Make every required subprocess and HTTP assertion propagate failure.

### Keep only if the corresponding public feature remains documented

- Prefix cache validation:
  `scripts/prefix_cache_canary.py`,
  `scripts/prefix_cache_eviction.py`, and
  `scripts/prefix_cache_latency_bench.py`.
- SM120 AOT export tooling:
  `scripts/export_gdn_sm120_aot.py` and
  `scripts/export_rms_fp4_sm120_aot.py`. These require an explicit locked Python
  environment; they are specialized build tools, not general user quick-start
  scripts.

### Archive as research or developer tooling

- Efficiency/performance studies, including
  `run_benchmark_efficiency*.sh`, `long_prompt_sweep.py`, profile, sweep, study,
  and summarizer scripts. They are useful research tools but several warn and
  continue after failures, so they are not release gates.
- OpenHands/SWE/capture tooling:
  `run_openhands_swebench.py`, `run_swe_kvmem_compare.py`,
  `swe_request_logits_parity.py`, `openai_proxy_capture.py`,
  `opencode_capture_server.py`, and `replay_openai_capture.py`.
- Top-level KVMem profiling, performance, and reproduction tools except the
  retained `kvmem_e2e_regression.py`.
- Attention and motivation experiment scripts.

### Exclude until repaired

- `scripts/plot_attention_step_kl_embedded.py`: zero bytes.
- Thirteen one-byte files under `scripts/motivation_v2/`.
- `scripts/run_swe_kvmem_compare.py` in its current form: it passes the deleted
  retrieval values `mean_attention` and `content_mean` to the current CLI.
- Any result JSON/JSONL currently visible as untracked files under
  `scripts/kvmem_eval/`.

The README also references the nonexistent
`scripts/mtp_acceptance_probe.py`; remove that reference or replace it with a
real, maintained test.

## KVMem evaluation suite

### RC0 recommendation: archive the directory as a unit

`scripts/kvmem_eval/` contains 180 tracked files: 70 Python files, 102 shell
scripts, one Markdown file, one JSON file, one JSONL file, and five text files.
At least 139 of those files contain machine-specific absolute paths.

The directory README identifies `run_eval.py` and `run_kvmem_eval.py` as the
canonical reusable entry points, but both default to untracked datasets, local
models, and local output directories. The many historical shell launchers are
frozen experiment provenance, not a portable public interface.

For a later research release, extract a small, separately tested package
containing only:

- `README.md`.
- `client.py`.
- `dataset.py`.
- `prompt.py`.
- `judge.py`.
- `grade.py`.
- `run_eval.py`.
- `run_kvmem_eval.py`.
- An explicit dependency file, input schema/sample, and environment manifest.

Before publishing that package, remove all machine-specific defaults, make
model/data/output paths explicit, document API-key handling, and add a
fresh-clone test. Historical launchers, ID lists, result summaries, and local
datasets should remain in the research archive.

## Benchmarks

### Archive

The six tracked files under `benchmark/mini_swe_deepswe/` are a recent research
harness, not an RC0 product gate. Its runner requires a local virtual
environment, local model tree, ignored DeepSWE checkout, and untracked runtime
and task files. It cannot run from a fresh clone.

The untracked `benchmark/claude_deepswe_ab/` material belongs with that archive,
not the RC0 product tree.

### Broken accuracy gate

The two local lm-eval YAML files are empty or one byte, ignored, and untracked.
`run_benchmark_accuracy.sh` runs an incomplete subset and converts lm-eval
failure into success. Until real pinned tasks, model configuration, expected
metrics, and failure propagation are added:

- do not call it a release gate;
- archive `scripts/run_benchmark_accuracy.sh`;
- exclude the empty benchmark YAML placeholders; and
- make `docs/benchmark_suite.md` describe only working checks.

### Exclude

- `benchmark/**/recordings/`.
- `benchmark/**/results/` and local summaries/states.
- `benchmark/deep-swe/` and other nested checkouts.
- Local benchmark virtual environments, services, sockets, and logs.
- Private or licensed datasets not explicitly cleared for redistribution.

## Documentation

All public KVMem documentation must label the feature **Experimental** until a
pinned release profile passes the real-model GPU gate.

### Keep

- `docs/architecture.md`.
- `docs/harness_compatibility.md`.
- `docs/kvmem_incremental_session_api.md`.
- `docs/kvmem_semantic_group_retrieval.md`.

### Keep after repair

- `README.md`: replace the performance-first structure with positioning,
  prerequisites, one tested GPU quick start, a small KVMem example, support
  status, and documentation links. Remove private paths, stale retrieval values,
  the missing script reference, and broken `DEVELOPMENT_LOG.md` links.
- `docs/release_baseline.md` (currently untracked): add it after updating the
  snapshot and real GPU evidence.
- `docs/benchmark_suite.md`: remove broken gates and refer only to tracked,
  runnable inputs.
- `docs/lm_eval_harness.md`: remove the nonexistent/empty “official-style”
  configurations or replace them with real pinned configurations.
- `docs/openhands_swebench.md`: add a redistributable instance sample/generator
  or remove its missing-file commands.
- `docs/claude_code.md`: move the DeepSWE operations section to the research
  archive unless its dependencies are published.
- `docs/kvmem_context_archive_design.md`: update “implementation in progress”
  into an Experimental current-interface specification.
- `docs/kvmem_milestone_checkpoint_benchmark.md`: split current checkpoint API
  documentation from historical benchmark results.
- `docs/kvmem_tiered_io_design.md`: retain ownership/mechanism documentation and
  move machine-specific empirical results to the archive.
- `scripts/kvmem_eval/README.md`: publish only with the later portable research
  package described above.

### Archive

Move these out of stable navigation and add a “frozen historical snapshot; not
a current support statement” banner:

- `benchmark/mini_swe_deepswe/README.md`.
- `docs/BUG_SUMMARY_large_ctx_garbage.md`.
- `docs/DEVELOPMENT_LOG.md`.
- `docs/KV_Memory_Paper.md`.
- `docs/kvmem_context_archive_status.md`.
- `docs/kvmem_known_issues.md`.
- Date-stamped KVMem optimization, performance, acceptance, and result pages.
- MemoryAgentBench, AgentLongBench, RAG, and paper-latency result pages.
- DeltaNet design/experimental pages.
- Superseded NVMe/SSD and paged-KV design proposals.
- `docs/kvmem_utility_eval_plan.md`.
- `docs/motivation_experiment_summary_en.md`.
- `docs/paged_kv_continuous_batching_plan.md`.
- `docs/paper.md`, `docs/paper_utility_efficiency_table.tex`, and
  `docs/section_43.tex`.
- `docs/kvmem_pipeline_timeline.html`.
- Untracked `docs/kvmem_deepswe_2m_extension_20260821.md`.

The exact currently tracked Markdown paths in this class are recorded in the
appendix.

### Exclude

- `docs/kvmem_deltanet_retrieval_implementation.md`: mojibake and an obsolete,
  disabled CLI path presented as current.
- `docs/kvmem_utility_evaluation.md` and
  `docs/kvmem_utility_evaluation_registry.json`: generated private result ledger
  containing thousands of absolute local paths.
- `docs/motivation_42.md`: incomplete experiment whose key scripts are empty and
  whose result assets are not tracked.

## Local assets excluded from RC0

Exclude these paths by directory, not only by filename extension:

- `models/`, including visible tokenizer/config/model-card files. Publish a
  model manifest and download instructions instead. If any tokenizer snapshot
  is later vendored, review its source, revision, license, and necessity first.
- `results/`.
- `build/`, `build-cuda/`, and `build-*/`.
- `.venv/`, `.venv-*`, and other local environments.
- `.claude/settings.local.json` and other machine-local editor/agent state.
- `third_party/openhands-benchmarks/` and all other nested repositories not
  explicitly vendored and audited.
- Caches, logs, sockets, database/WAL files, recordings, generated manifests,
  local service state, and private datasets.

The current `.gitignore` ignores model weight extensions but leaves 29 model
metadata/tokenizer files visible, totalling roughly 46 MiB. RC0 should ignore
`models/` as a whole and explicitly allow only a separately created public
manifest outside that directory.

The current benchmark ignore rule also conflicts with six already-tracked
MiniSwe files. Before the release candidate is created, either add an explicit
audited allow-list or remove those files from the public tree. The command
`git ls-files -ci --exclude-standard` should have no unexplained output.

## P0 blockers before a public candidate commit

1. Record the four intended tracked deletions and add
   `tests/prompt_render_test.cpp`; otherwise the clean tree does not build the
   code that was tested in the working tree.
2. Track the Apache-2.0 license, third-party notices, and the two missing MIT
   license texts.
3. Close `.gitignore` around `models/`, results, recordings, local environments,
   build variants, secrets, and local agent/editor state.
4. Decide whether MiniSwe and KVMem evaluation material is excluded or moved to
   a separately labelled research archive.
5. Remove empty scripts, untracked result data, stale CLI values, missing-file
   references, and private absolute paths from the public tree.
6. Pin the build dependency stack and one model source/revision/file/SHA-256,
   including the model license and download or conversion procedure.
7. Resolve the code-provenance questions listed above.
8. Build a strict real-model GPU release smoke; do not rely on host tests or
   no-device CUDA false-positive passes as runtime evidence.
9. Run a pinned secret scanner on the exact final tree and, if history is kept,
   on every object reachable from the exact public ref.
10. Generate the release archive only from the final audited commit tree and
    reproduce build/tests from that archive.

## Proposed change batches after approval

No batch below should begin until the owner approves the boundary and history
strategy.

### Batch A: public-tree boundary

- Update `.gitignore` to exclude whole local asset categories.
- Add licenses/notices and the prompt-render test.
- Record the backend/mock deletions already present in the working tree.
- Remove or relocate the explicitly excluded and archived material from the
  public tree without deleting it from the private research repository.
- Produce a clean candidate index and review `git diff --cached --stat` and
  `git diff --cached` before committing.

### Batch B: reproducible GPU profile

- Pin CUDA/compiler/FlashInfer/CUTLASS/CCCL and one model artifact.
- Add a strict, short GPU release smoke and provenance artifact.
- Fix false-green CUDA test behavior.
- Run it twice from the same clean exported snapshot.

### Batch C: CI and governance

- Add required host build/unit checks.
- Add path/link/license/secret/large-file hygiene checks.
- Add a protected self-hosted GPU workflow or document a manual release GPU
  gate if no trusted runner exists.
- Add contribution, security, support, and conduct documents.

### Batch D: README and stable documentation

- Replace the root README with the verified public onboarding path.
- Generate the support matrix and commands from the pinned profile rather than
  duplicating them manually.
- Keep historical performance claims out of the quick start and attach every
  published result to a commit/profile/artifact.

## Decisions required from the owner

1. Approve the recommended sanitized public root commit, or explicitly choose
   to retain the audited single-branch history.
2. Approve excluding MiniSwe, the current `scripts/kvmem_eval/` tree, historical
   experiment scripts, and dated result documents from RC0.
3. Decide whether the first release promises plain native CUDA inference only,
   or requires one KVMem path to be release-tested. The proposed default is
   plain native CUDA plus one small Experimental KVMem demonstration.
4. Name the single public model artifact and repository/revision that may be
   used in the release gate.
5. Confirm the code provenance and contributor rights for the `qw3_ly`,
   tokenizer, and FlashInfer-related areas.
6. Provide the public security contact and repository/organization name.

## RC0 boundary acceptance criteria

The boundary phase is complete only when:

- the public index contains no build, model, result, recording, environment,
  session, database, or private dataset assets;
- every tracked executable script has tracked inputs or explicitly documented
  external inputs;
- public docs contain no machine-private paths and no broken local links;
- no empty placeholder script is presented as runnable;
- all shipped source has a known project or third-party provenance;
- license and third-party notice files are present and consistent with the
  actual vendored/build source;
- a formal secret/large-file scan passes on the exact public tree/history;
- `git archive <RC_SHA>` builds and passes all host tests in a clean environment;
  and
- the subsequent pinned real-model GPU gate passes before any CUDA runtime path
  is marked Tested.

## Appendix: exact Markdown archive set

- `benchmark/mini_swe_deepswe/README.md`
- `docs/BUG_SUMMARY_large_ctx_garbage.md`
- `docs/DEVELOPMENT_LOG.md`
- `docs/KV_Memory_Paper.md`
- `docs/kvmem_context_archive_status.md`
- `docs/kvmem_known_issues.md`
- `docs/kvmem_adaptive_index_optimization_20260730.md`
- `docs/kvmem_adaptive_index_placement_ab_20260730.md`
- `docs/kvmem_assembly_rerope_optimization_benchmark_20260724.md`
- `docs/kvmem_chunked_reselection_optimization_20260807.md`
- `docs/kvmem_claude_serving_bugfix_acceptance_20260820.md`
- `docs/kvmem_cpu_proactive_writeback_benchmark_20260725.md`
- `docs/kvmem_cpu_transfer_optimization_benchmark_20260724.md`
- `docs/kvmem_fp8_performance_benchmark_20260725.md`
- `docs/kvmem_generic_performance_plan_20260725.md`
- `docs/kvmem_gpu_memory_optimization.md`
- `docs/kvmem_memoryagentbench_archive_evaluation.md`
- `docs/kvmem_memoryagentbench_baselines.md`
- `docs/kvmem_performance_evaluation_20260726.md`
- `docs/kvmem_rag_agentlongbench_results.md`
- `docs/kvmem_rag_retrieval_overlap.md`
- `docs/kvmem_semantic_stagein_optimization_20260801.md`
- `docs/kvmem_ssd_writeback_benchmark_20260724.md`
- `docs/kvmem_storage_optimization_benchmark_20260723.md`
- `docs/mtp_paged_kv_optimization.md`
- `docs/paper_pre_answer_latency_completion_plan.md`
- `docs/paper_pre_answer_latency_results_20260804.md`
- `docs/paper_pre_answer_latency_results_20260812.md`
- `docs/results.md`
- `docs/kvmem_deltanet_retrieval_design.md`
- `docs/kvmem_deltanet_retrieval_experimental.md`
- `docs/kvmem_implementation_notes.md`
- `docs/kvmem_nvme_ssd_architecture.md`
- `docs/kvmem_ssd_complete_design.md`
- `docs/kvmem_utility_eval_plan.md`
- `docs/motivation_experiment_summary_en.md`
- `docs/paged_kv_continuous_batching_plan.md`
- `docs/paper.md`
