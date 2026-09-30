# Sampling contract and same-GPU reference qualification

The goal is active. This stage qualifies the public reference and aligns the
sampling contract; it does not establish a public throughput result.

## A real distribution difference

At pinned SGLang revision `9f6fc553322e897a22f30320dae750f4a40a0b84`,
[`build_dflash_verify_target_probs`](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/speculative/dflash_utils.py)
applies top-k, normalizes those candidates, then applies top-p. Its ordinary
[`Sampler`](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/layers/sampler.py)
uses joint top-k/top-p filtering by default when min-p is disabled. QW3's
existing ordinary and speculative samplers also use joint filtering.

For probabilities `[.4, .3, .2, .1]`, top-k 2 and top-p .55, joint filtering
retains the first two candidates; top-k-first retains only the first. Identical
temperature/top-k/top-p numbers alone do not establish identical distributions.
This difference must not be hidden in a throughput comparison.

QW3 now exposes `--sampling-filter-order joint|top-k-first`, also available as
`sampling_filter_order` in HTTP requests and `GenerationOptions`. The default
remains joint. The option applies consistently to ordinary decode, built-in
MTP, DFlash, continuous ordinary/MTP sampling, and the archive/session wrappers.
Top-k-first also keeps all candidates tied at the nucleus cutoff, matching
FlashInfer's top-p operator. Exact top-k boundary tie selection remains kernel
dependent across engines; no bitwise cross-engine tie-order equivalence is
claimed. The rejection step always uses the same configured target distribution
as ordinary QW3 generation and the actual conditional draft probabilities.

The explicit comparison policy for SGLang DFlash and **all three QW3 methods**
will be top-k-first. Any ordinary SGLang result using joint filtering must be
labeled separately; it is not a same-distribution autoregressive control for
that DFlash configuration. Upstream inference code is unmodified.

## Evidence

`qw3-target-sampling` checks 1,444 analytical/random cases against a full-sort
long-double oracle, including top-k-only, full vocabulary, temperature, min-p,
nucleus filtering and cutoff ties. The upstream fixture contains 60 cases,
including the actual 248320-token vocabulary. Both candidate support and
probabilities pass (absolute probability bound 2e-6).

`scripts/dflash_sampling_fixture.py` verifies SHA-256 of both pinned SGLang
source files and extracts their unmodified function ASTs, avoiding serving
initialization. The DFlash fixture executes the real FlashInfer 0.6.18 CUDA
renormalization operators; the joint fixture executes the independent SGLang
PyTorch sampler. Raw inputs, probabilities and source hashes are in
`benchmark/dflash2/sglang_sampling_fixture_v2.json`.

```bash
cmake --build build-dflash-host --target qw3-target-sampling -j 4
build-dflash-host/qw3-target-sampling benchmark/dflash2/sglang_sampling_fixture_v2.json
```

## Reference environment

SGLang source is pinned at the revision above, installed in the isolated
`benchmark/dflash2/sglang-env-py312` environment: Python 3.12.13, PyTorch
2.13.0+cu130, Transformers 5.12.1, FlashInfer 0.6.18. The original QW3 Python
environment is preserved. Full package versions are in
`sglang_environment_lock.txt`; source/launch records are retained alongside it.
The supported `SGLANG_BUILD_RUST_EXTS=none` install setting was used; the default
Python radix-tree backend is unchanged. An initial Python 3.13 attempt failed
building outlines-core and is retained in the installation logs.

`scripts/dflash_reference_server.py` verifies the source revision and clean
inference files, writes the exact launch manifest, and launches the server with
CUDA graphs and FlashInfer attention on SM120. Environment PATH includes its
own tools and CUDA 13 for JIT compilation.

Two comparison conditions remain open before formal timing:

- The historical reference qualification uses BF16 target weights and KV.
  The user's current target is Q8_0 weights + FP8 KV; new native BF16 target/KV
  support has been removed at their request. The historical reference is not a
  matched-precision baseline for this deployment. Qualify the reference's
  quantization/quality conditions explicitly before making a performance claim.
- The initial 32-request configuration is capped by SGLang's FP32 recurrent
  state cache to six active requests (34 cache slots, five slots/request).
  Freeze an appropriate memory configuration for this GPU before formal runs;
  record both client concurrency and actual server admission capacity.

Frozen public prompts, revisions and output limits remain unchanged. Neither
installation nor a successful synthetic smoke request substitutes for the
five-task, three-concurrency, repeated performance and quality gates.

The initial server completed qualification: two identical greedy 64-token
requests produced identical output, and eight concurrent sampled 64-token
requests completed successfully. CUDA graphs remained enabled for both models
and the selector. `sglang_smoke_01.json` and `sglang_server_01.log` retain the
responses and runtime configuration. The service was then stopped cleanly to
free the GPU for native regressions.

The completion API ignores `return_meta_info`; request
`return_spec_tokens_details=true` to obtain acceptance counters in
`sglext.spec_tokens_details`, and `return_token_ids=true` for exact output IDs.
This was verified by `sglang_completion_metrics_probe.json`. Future raw-prompt
benchmark clients must use these fields, rather than silently losing the
acceptance metrics expected by the official chat benchmark.

All 18 local target shards were SHA-256-checked against their Hugging Face
download metadata. Target revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; per-file hashes, sizes, revisions
and tokenizer/config files are frozen in `target_bf16_manifest.json`.
