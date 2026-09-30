# DFlash 2 / QW3 / KVMem execution goal

Created: 2026-09-24. Status: **active**.

Latest user steering (2026-09-25): [long-running single-request acceptance](dflash2_long_running_acceptance_20260925.md)
now controls the transition to concurrency: wait for an idle GPU, validate real long-history
DeepSWE-like decode near 130+ tok/s and total GPU memory below 48GB, retain correctness
checks, then stop single-request speed tuning and proceed to concurrency. The older
requirement to chase H200 absolute throughput before that transition is superseded.

Current acceptance contract: [Q8_0 + FP8 KVMem goal](dflash2_q8_fp8_goal_20260924.md).
The user explicitly recreated the persistent goal with this scope. That contract
supersedes older scope and acceptance wording below; historical progress remains
preserved here.

Work in progress: [joint DFlash proposals](dflash2_joint_draft_20260925.md)
batch draft projections, convolution, independent-cache attention and selection.
Operator/fixture and initial C4 evidence passes; the corrected draft/verification
rendezvous is undergoing wider KVMem validation. This remains opt-in.

Current optimization experiment: [wide Q8 verification](dflash2_wide_q8_verifier_20260925.md).
The 32-row candidate passes operator parity and memcheck and improves the tested
warm projection shapes. Another 192 KVMem responses pass direct parity with the
reference, and CUDA racecheck reports zero hazards. It remains opt-in;
end-to-end performance qualification remains open.

Latest work — 2026-09-25: [DFlash continuous KVMem controller](dflash2_continuous_controller_20260925.md)
preserves each cold request's full control flow while jointly executing target
verification, feature capture and checkpoint/logit transport. Scoped C4/C16
DFlash KVMem and dense probes pass; draft proposals still execute individually.
Private query completion and longer refresh pass. Queued replenishment exposed
batch-dependent Q8 arithmetic; retaining ragged verification and decode-compatible
Q8 tiles passes its controlled repeat test, while disabling the tiles reproduces
the failure. The final default matrix passes 448 responses (416 KVMem), with
1,611 actual joint DFlash target batches and zero GPU-sampling fallbacks.
Single-request none/MTP/DFlash regressions pass all 24 CPU/GPU pairs (48
responses), and 34 host tests pass. Shared prefix/archive/session/repository
lifecycle, serial/CB numerical quality, joint draft optimization, formal
independent/public performance and final integration remain open.
The preceding [shared KVMem pools](dflash2_shared_kvmem_pool_20260925.md),
[continuous sampling](dflash2_continuous_sampling_20260925.md) and
[GPU sampling/FP8 correction](dflash2_gpu_sampling_20260925.md) retain their
historical scope and per-run binary identities.

## User priority update — 2026-09-24

Latest precision constraint: **Q8_0 target weights + FP8 KV**. The user explicitly
asked to remove the newly added BF16 target-weight and BF16-KV support. Those
extensions and their active test targets have been removed. BF16 target loading
and BF16 KVMem storage are no longer deliverables. Keep the official DFlash
draft's BF16 weights and the framework's pre-existing BF16 arithmetic (including
internal Q8 residual storage). Historical BF16 results below are archived
experiments, not current supported configurations or deployment baselines.

The user clarified: “核心是要支持kvmem，普通qw3可以仅仅是测试通过就好了”.
KVMem is the primary implementation and performance deliverable. Ordinary QW3
must pass correctness/regression checks and remains a useful control; achieving
an independent ordinary-QW3 performance target is no longer required. Prioritize
FP8 KV, retrieval/replay/resume/archive/compaction, bounded state and DFlash
continuous batching inside KVMem. Evaluate end-to-end request throughput,
latency and memory, alongside decode throughput. Public reference comparisons
remain required evidence for the requested KVMem performance goal, with context
and precision differences explicit; a short ordinary-QW3 speedup cannot satisfy
the goal. This explicit user update supersedes the earlier equal performance
priority for ordinary QW3 and KVMem.

## Outcome

Run the complete Qwen3.8-27B DFlash 2 algorithm efficiently in KVMem, with
correct target sampling and transactional hybrid-model state, and maintain
ordinary QW3 correctness. Reach and exceed reproducible performance baselines
for the KVMem goal, using public implementations as explicitly qualified
references. A runnable prototype alone does not complete this goal.

Implementation worktree: `/home/chaidi/qw3-dflash2`, branch
`codex/dflash2-qwen38`. Base HEAD is
`4954b871aa37b3e8530aef1cfaade2c2bc3d60f3`, plus the existing worktree's runtime
changes captured in `benchmark/dflash2/workspace_base.json`. Those inherited
changes belong to the prior workspace and must not be mistaken for DFlash work.
Original workspace files remain available for integration; build and experiment
outputs are isolated. Model files are shared under `/home/chaidi/qw3/models`.

## Acceptance contract

1. **Correctness:** greedy output and resumed state match the same target with
   speculation disabled, subject to diagnosed numerical parity; stochastic
   acceptance/residual sampling preserves the exact configured target
   distribution. Exercise all rejection boundaries, short blocks, EOS, stop,
   repetition/presence penalties, cancel/resume, and request isolation.
2. **Complete draft:** five-layer feature capture, non-causal sliding attention,
   grouped dynamic convolutions and selector; no omitted DFlash 2 modules.
3. **KVMem:** pass warm continuation, prefix fork, selection/rebase, archive,
   thinking compaction and bounded-cache tests, relative to the same KVMem
   policy without speculation. Draft history stays bounded.
4. **Public comparisons:** lock target/draft revisions, quantization, engine
   revisions, prompts, rendering/reasoning, sampling, output limits and timing
   definitions before final runs. Reproduce public implementations on the same
   RTX PRO 6000 Blackwell. Compare the KVMem none/MTP/DFlash routes under the
   same retrieval policy, and qualify the public reference against that workload;
   do not switch precision or workloads between contestants. Ordinary QW3 is a
   correctness/control route, not an independent performance acceptance target.
5. **Performance:** on the fixed KVMem task matrix, target at least 5% greater
   aggregate throughput (geometric mean of paired ratios), with every required
   cell reaching baseline within measurement uncertainty and no hidden large
   latency/quality regressions. Use at least three independent runs, alternate
   engine order, and report confidence/dispersion; near-threshold results require
   more evidence. This is an engineering success criterion, not an assertion of
   achieved performance.
6. **Published H200 results:** pursue and report the published raw throughput,
   acceptance lengths and relative speedups separately, with hardware and
   precision differences explicit. Record reached/missed per cell; same-GPU
   comparisons cannot be relabeled as reproduction of H200 absolute numbers.
7. **KVMem performance:** predefine multi-turn long-history workloads and working
   set sizes; compare the same retrieval policy with none/MTP/DFlash 2. Report
   retrieval/prefill and decode separately. Full-context public engines provide
   context, not an identical-semantics KVMem correctness oracle.
8. **Reproducibility:** preserve commands, manifests, raw samples, phase timings,
   VRAM, correctness checks, and result summaries. Ship the implementation and
   tests into the user's intended workspace/branch after validation. Do not mark
   the goal complete while an implementation or mandatory performance gate is
   still unresolved.

## Fixed initial public protocol

Source: [official DFlash 2 model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4).
Target: Qwen3.8-27B. Draft revision:
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Block size 8 (7 draft tokens).
Temperature 1, top-p .95, top-k 20, reasoning effort xhigh, maximum 4096 new
tokens. Tasks: GSM8K, MATH-500, HumanEval, MBPP, MT-Bench. Concurrency 1, 8, 32.
Throughput is actual total completion tokens divided by end-to-end elapsed time.
Use the pinned upstream benchmark prompt formatting; freeze exact dataset
versions/sample IDs before performance tuning. Evaluation split is not used to
tune thresholds or select favorable prompts.

Frozen input artifact: `benchmark/dflash2/public_workload_v1/manifest.json`.
Official benchmark revision `07ebd93db9f472af339b644bb70221ad8428328a`;
128 measured requests/task plus the next 1/8/32 shuffled rows for warmup
(shuffle seed 42, repeat when the dataset is shorter). The serving benchmark
uses only the first MT-Bench turn. The frozen 800 input records, including
warmup, contain 107,581 token IDs; all match the native tokenizer exactly.
Use the same pre-rendered xhigh prompt/token IDs on both engines, since the
current native chat endpoint does not apply the upstream reasoning-effort
template contract. Repeated public timing remains open.

Reference qualification subsequently passed on the same GPU with CUDA graphs:
two matching greedy requests and eight concurrent sampled requests. The public
task matrix has not run. The pinned reference DFlash verifier uses top-k-first
filtering; all native comparison methods must explicitly use that policy.
See `docs/dflash2_sampling_reference_20260924.md` and
`benchmark/dflash2/public_execution_protocol_v1.json`. Matching target KV dtype
and freezing the server memory/admission configuration remain prerequisites.
All 18 BF16 target shards match their HF SHA-256 metadata, at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Additional native matrix: contexts 4K/32K/128K, concurrency 1/4/16; KVMem history
and working-set sweeps; greedy and stochastic tests. Results on this matrix
must be labeled separately from the public protocol.

## Milestones

- [ ] M0: frozen model/dataset/engine manifests and verified same-GPU reference.
- [ ] M1: loader, target feature capture, complete single-request CUDA draft,
  greedy verification and hybrid-state rollback.
- [ ] M2: general sparse proposal sampling, numerical/distribution checks, GPU
  top-k/selector and measured phase costs.
- [ ] M3: KVMem/prefix/recovery integration, bounded draft state and regressions.
- [ ] M4: continuous batching and fair public/native/KVMem performance sweeps.
- [ ] M5: optimize remaining bottlenecks, repeat required gates, integrate code
  and publish reproducible local report. Website or remote publishing is not
  required.

## Progress log

2026-09-25, Q8/FP8 KVMem archive stage:

- Persist and restore a bounded, fingerprinted DFlash raw-K/V window with each
  target ladder. Same target archives support target-only and DFlash readers;
  legacy or different-draft caches explicitly cold-start.
- Target-only/DFlash 4K archives have identical target KV, token and state bytes.
  A resumed 4K→8K build matches uninterrupted target and draft ladder files.
- Direct, residual and pre-first-ladder reloads pass greedy controls. Fixed
  explicit no-MTP mode being overridden and zero-ladder replay preserving a
  nonexistent source index. Guided fallback, completed 27-token compact queries
  and attention-probe policies match. Successful matrices total 20 processes
  and 32 responses.
- 33 host tests, exact GPU draft archive continuation and zero-error memcheck
  pass. See `docs/dflash2_kvmem_archive_20260925.md` for per-run evidence and limits.
- The goal remains active: GPU target sampling, actual DFlash continuous
  batching, remaining advanced/quality routes, formal public/performance
  comparisons and final integration are still open.

2026-09-24, Q8/FP8 KVMem lifecycle stage:

- Fixed the attention merge's row-dependent reduction dispatch. All widths
  1–8 at four post-4K positions now match target residuals and logits exactly;
  168 attention serial/batch configurations pass.
- 4K/32K histories pass strict production comparison with a fixed 8K bounded
  repository, including cancellation, old-prefix changes and reclamation.
- Draft snapshots survive cloning and CPU offload/restore. Twenty compaction
  transaction executions preserve target/draft/query bytes, including a 4095
  boundary beyond the draft window.
- Advanced target-only KVMem requests now have a no-MTP control route. Fixed
  false warm-checkpoint rejection and speculative refresh-boundary overrun;
  generated private queries and refresh policy now match the serial route.
- Archive, production summary quality, advanced bounded APIs, GPU sampling,
  continuous batching, formal comparisons and final integration remain open.
  Evidence: `docs/dflash2_kvmem_lifecycle_20260924.md`.

2026-09-24, Q8/FP8 arithmetic qualification:

- Fixed convolution accumulation order, batch-dependent MMVQ reductions and
  batch-8 verifier quantization; fused tiny Q8 SwiGLU. Widths 1–8 now pass the
  target-only strict residual/logit equality probe at four positions.
- Math/code/Chinese 64-token target integration, seven rollback boundaries,
  snapshots, 44 ordinary and 32+32 normal/bounded KVMem production checks pass.
- Added a verifier operator regression which exposed a shared-memory DeltaNet
  race. Broadcast barriers fix the four affected fallback variants; racecheck
  and memcheck now report zero errors. Six selected CUDA CTests pass.
- Q8 upstream file identity verified against Unsloth revision
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`.
- Matched-output 256-token warm KVMem diagnostic: end-to-end 44.22 plain versus
  87.38 DFlash tok/s (1.976×), decode 44.87 versus 90.36. These are three
  same-process warm repeats, not independent formal performance acceptance.
- Long-history, archive/compaction, GPU sampling, DFlash continuous batching,
  public qualification, repeated matrices and final integration remain open.

2026-09-24:

- Created active goal and isolated worktree; recorded inherited source hashes.
- Downloaded pinned BF16 draft checkpoint (3,848,817,896-byte safetensors file).
- Added separate DFlash config/tensor validation and raw standalone/sharded HF
  source. `qw3-dflash-model` and the existing `qw3-small-model-config` host
  regressions passed. The actual checkpoint audit bound all 81 tensors and
  1,924,404,480 parameters successfully. No GPU draft implementation or
  performance claim yet.
- Recorded config and weight SHA-256 in `benchmark/dflash2/draft_manifest.json`.
- Existing Python environment has torch 2.11.0, transformers 5.10.2 and vLLM
  0.22.0, but its vLLM package has no DFlash 2 model module. Use an isolated
  pinned reference environment/build; do not mutate the existing environment.

- Added GPU weight binding and a complete standalone draft executor: grouped
  dynamic convolutions, noncausal sliding attention, five backbone layers,
  target-shared output head, GPU top-k and conditional selector. The initial
  implementation keeps activations and draft KV in FP32 (about 80 MiB of KV at
  the official window); BF16 KV and kernel optimization remain future work.
- Pinned official PyTorch source generated seven small FP32 full-network
  fixtures, including empty prefixes, window overflow, short blocks and nonzero
  positions. Native CUDA outputs agree within **1.07288e-6** maximum absolute
  error and all greedy proposals agree. One-shot/chunked prefix appends and
  repeated speculative calls pass. The actual BF16 checkpoint also executes
  all five layers with finite, repeatable outputs on synthetic features.
- CUDA operator tests passed for convolution, attention, 248320-vocabulary
  top-k, FP32/BF16 selector and FP32 reference linear algebra. Compute Sanitizer
  memcheck reported zero errors and racecheck zero hazards. Feature capture
  subsequently added FP32/BF16 strided packing tests, both exact.
- Added opt-in post-residual target taps and a model-dependent native generation
  probe. Q8 target plus BF16 DFlash matches serial greedy on the first two
  64-token prompts (6.4 and 5.82 committed tokens/cycle). The third Chinese
  prompt diverges at output index 61. This is an **open correctness gate**;
  logit diagnostics are in progress. No public benchmark gate has passed.
- Host loader/config regressions still pass after the executor additions.
- Added native unquantized HF target loading, retaining the existing target
  norm/recurrent tensor transformations and rejecting unsupported quantization
  methods. This permits a BF16 target comparison without modifying model files.
- **BF16 target integration passed:** three 64-token prompts match serial greedy
  exactly. Their committed tokens/cycle are 6.4, 5.82 and 2.56. These are short
  diagnostic prompts, not the frozen public task matrix or a speedup result.
- All seven rejection boundaries (keep 1..7 target input rows) pass for both Q8
  and BF16: changing every rejected suffix token leaves accepted features and
  the next three continuation logits **bit-identical**.
- Q8 discrepancy isolated: replaying the same 23 verifier blocks without any
  draft calls reproduces every saved bonus-logit vector bit-for-bit, including
  the Chinese output divergence. This rules out draft interference for this
  case; serial/batched target numerical parity remains an open gate. Simply
  increasing `QW3_MATMUL_MMA_MIN_BATCH` to 9 did not resolve it.
- Added BF16-weight fixtures with the native GEMM input precision modeled on
  the official topology. Greedy proposals agree in all seven cases. FP32
  maximum error remains 1.07288e-6; BF16 maximum error is 0.00318742 and maximum
  relative L2 error 0.000756959. BF16 input rounding requires a separate
  precision bound; these results must not be reported as FP32-level parity or
  as equivalence to an upstream all-BF16 activation implementation.
- Sparse rejection-sampling primitive passes six analytical distribution
  checks and 400,000 two-token conditional trials (maximum standardized
  deviation 1.397). It is not yet wired to native generation, so stochastic
  end-to-end correctness remains outstanding.
- Extended CUDA feature-packing tests passed memcheck (zero errors) and
  racecheck (zero hazards). DFlash tap capture currently disables the separate
  continuous-body route until that route has its own per-layer capture hooks.

- Added uint64 logical draft positions with bounded RoPE frames rebased from
  raw keys; three rebases and 16 snapshot continuations pass in both fixtures.
  The raw-key cache increases persistent draft KV to approximately 120 MiB per
  request. Paired target/draft snapshot restoration is exact on the real model.
- Connected `--dflash-model` to native greedy and stochastic generation,
  including correct EOS verification, cancellation recovery and thinking-budget
  boundaries. Dense production probes pass 18 cases on each plain/DFlash route
  in both the FlashInfer and non-FlashInfer builds.
- Connected KVMem query replay and paired draft state. Normal and bounded
  repository production probes pass 24 checks each, covering sparse selection,
  repeated prefix, tail rewrite, EOS, cancellation/retry and seeded sampling.
  Fixed three baseline KVMem issues exposed by these tests: replay position
  selection, stale checkpoint admission, and released GPU index buffers after
  switching from a sparse history to a cold request. Failed runs are retained.
- Froze five official serving workloads, dataset revisions and xhigh rendering;
  all 800 input records / 107,581 token IDs match the native tokenizer exactly.
  At that stage, same-GPU reference execution and public timing had not run.
  The later reference qualification is recorded below; public timing remains open.
- Qualified pinned SGLang in an isolated Python 3.12 / Torch 2.13 / FlashInfer
  0.6.18 environment on the same GPU with target/draft/selector CUDA graphs.
  Two greedy repeats and eight concurrent sampled requests pass. Verified all
  target shard SHA-256 hashes and froze target revision. Public timing remains
  open; the initial reference memory configuration admits six active requests.
- Added explicit joint/top-k-first sampling policy consistently across all
  native generation modes; 1,444 analytical/random cases and 60 pinned upstream
  distribution cases pass, including nucleus ties. Latest production probes
  pass dense 44, normal KVMem 32 and bounded KVMem 32 checks.
- Fixed raw completion streaming (previously ignored `stream=true`). Plain,
  built-in MTP and DFlash each pass 17 HTTP checks for token/string prompts,
  both policies, streaming/usage, stop, UTF-8 and disconnect/retry. All six
  512-token disconnect probes stop after one emitted token. Host CTest passes
  all 32 cases. Advanced KVMem HTTP/session/archive routes remain unvalidated.

- Added native BF16 target/MTP KV allocation, append and FlashInfer dense/paged/
  ragged attention. Six attention CTests pass, including FP16/FP8 regressions;
  BF16 memcheck reports zero errors. Real rollback and paired snapshots pass.
  Math/code greedy prompts pass; Chinese output differs from serial target at
  index 41 and exactly reproduces under pure-target batch replay. Disabling
  small-pair GEMV and enabling local paged KV do not remove it. The gate remains
  open, as does KVMem BF16 remapping/storage.
- BF16 KV HTTP qualification passes 85 checks across dense plain/MTP/DFlash
  and continuous plain/MTP. Both continuous routes complete eight concurrent
  requests with actual body/verifier batch 4. DFlash is still serialized; these
  functional probes do not establish a public throughput or quality result.

Next, with KVMem as the primary deliverable: qualify the existing Q8_0-weight /
FP8-KV deployment, validate archive/guided replay, thinking compaction and
long-history GC; optimize warm checkpoint reuse, GPU target sampling and DFlash
continuous batching. Keep weight precision explicit: the BF16-weight ~21 tok/s
short plain diagnostic is not the prior Q8_0 baseline. Corrected plain timing
separates post-prefill selection/replay from decode, as MTP/DFlash already do.
Close Q8_0/FP8-KV target parity, freeze the isolated upstream reference
precision/memory/admission configuration, and run the repeated KVMem matrix.
BF16 target weights and BF16 KV are explicitly out of scope per the latest
user instruction. Ordinary QW3 needs correctness/regression qualification only.
No public performance gate has passed. See
`docs/dflash2_implementation_status_20260924.md` for reproducible commands.
