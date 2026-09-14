# Qwen3.5 9B / 4B adaptation and validation

Baseline commit: `726d149`. User authorized adaptation and tests on 2026-09-14.

## Acceptance

* Qwen3.5-9B and 4B load from Unsloth Q8_0 GGUFs with verified upstream SHA-256 and pinned revisions. Download via hf-mirror with HTTP_PROXY, HTTPS_PROXY and ALL_PROXY removed in both cases.
* Preserve FP8 KV, MTP chain 4, full model computation, KVMem mean-K selection, guided query and replay. Ordinary inference keeps native Claude Code compaction.
* Validate HF nested RoPE configuration and dynamic MTP layer mapping; instantiate GQA=4 BatchDecode without changing the GQA=6 path.
* Parameterize the GPU vision frontend for each model's own tower and output width; check numerical parity for the 64-dimensional vision heads and retain 27B parity.
* Verify short generation, MTP on/off greedy consistency, paged continuous batching, prefix reuse and selection/replay beyond the configured budget. Include a 27B regression.
* Run a fixed DeepSWE pilot: the first eight IDs of the existing 16-task list, seed=1000, both context policies for each smaller model. Do not select tasks based on pilot scores. Retain failures and raw traces, report accuracy and latency separately.

## Resource and configuration policy

Use a separate build and result directory. Existing OSWorld uses its frozen binary and files. Download and CPU build can proceed while OSWorld runs. GPU timing and complete agent trials require exclusive GPU ownership; pause at an arm boundary and resume the original suite afterward, preserving completed results.

Before pilot launch, record exact commands and environment, model hashes, task IDs, Claude Code version, and the matching 27B configuration. The current OSWorld reference is Q8_0 / FP8 / MTP=4, plain ctx=262144 and KVMem ctx=2097152, selection=131072, generation=131072, block=32, CPU history/index, NVMe=0, GC off. Historical DeepSWE has a different generation budget (65536); do not silently pool those measurements with the new pilot or claim a matched cross-model comparison if the budget differs.

## Evidence location

`benchmark/qwen35_small_20260914/` holds manifests, download status, model binding checks, numerical checks, throughput measurements and the pilot's raw traces. Only validated results are reported as completed. Static compatibility and successful compilation alone do not satisfy the goal.

## Qualification and pilot status (2026-09-14)

The text build is frozen at SHA-256
`d6cf84c7a3e244ce1667e83b8b6034f3506a78784d92c6f862721ad75fbcb4e3`.
21 CPU tests and six independent attention-oracle cases passed (GQA 4/6,
FP8/FP16). Both models completed 8K/128K generation, MTP, prefix cold/warm
restoration and actual semantic selection/query replay. Guided generation was
also exercised. The 27B text and vision regression passed. Free-running MTP
on/off outputs are not exactly equal under FP8 batch arithmetic; they must not
be described as an exact greedy-equivalence result.

The Unsloth main Q8_0 files do not include MTP. Official MTP tensors were
downloaded separately, converted with the recorded llama.cpp converter and
appended; every main-model Q8 tensor was checked byte for byte. Auxiliary
`*-BF16` directories contain only vision/MTP tensors, not complete BF16 language
weights. Download provenance records repository revisions, sizes and hashes.

Small-model GPU vision is not yet qualified: the existing cosine >0.999 gate
gave 0.998719 (4B) and 0.998071 (9B) on the selected screenshot, with small
initial rounding differences amplified across layers. This limitation does
not apply to the text-only pilot. Do not report these as passed vision tests
or start a small-model OSWorld experiment without resolving this gate.

The pilot copies the existing 27B DeepSWE profile: Q8_0, FP8 KV, MTP=4,
selection 131072, generation 65536, block 32, logical history 3145728,
CPU backing cap 110 GiB, guided query limit 4096, thinking cap 28672,
temperature 1.0/top-p 0.95/top-k 20. Mean-K index placement remains GPU,
as in that DeepSWE reference. Ordinary context and native auto-compaction
window are 262144. NVMe and GC are not enabled. The fixed first eight tasks,
seed 1000, two models and two policies give 32 trials. After the first 4B pair,
the user requested 9B first: all sixteen 9B arms now precede the remaining 4B
arms, retaining the original within-model counterbalanced policy order.
`pilot_plan.json` records this schedule revision and the original plan/source
are archived under `controller_revision_9b_first/`. Model binaries, parameters,
task membership and completed results are unchanged. Existing agent scripts, dependency/network
preflight and verifier images are reused. Infrastructure errors pause the
queue and task failures remain in the data.

Run `python3 scripts/run_qwen35_deepswe_pilot.py --prepare-only` to audit the
manifest, and `--run` to start/resume. `pilot_plan.json`, per-arm manifests and
`pilot_status.json` record the frozen artifacts and progress. OSWorld was
paused after 060 completed; all 50 completed arm-result hashes are recorded
in `pause.json` and must remain unchanged when resuming.

The first completed 4B pair (`superjson-error-stack-serialization`) had reward
0 in both arms: 0/80 feature tests and 116/116 existing tests passed. Both
agents and verifiers exited normally; retain these quality failures. Weighted
decode throughput was 128.74 token/s for ordinary qw3 and 125.53 token/s for
KVMem, over different trajectories (82,586 vs 101,100 decoded tokens). KVMem
performed one real guided refresh: 919.6 ms query generation, 141.5 ms
selection/KV organization, 301.5 ms replay and 2.027 s visible-content TTFT.
No aggregate ability or speedup claim is supported by this one pair.

An additional visual diagnostic (`vision_backend_audit/comparison.json`)
compared native BF16 output and PyTorch math/Flash BF16 output against a
PyTorch FP32 math reference, without changing the original pass threshold.
For 4B, relative RMS error against FP32 was 8.57% native, 9.23% PyTorch math
BF16 and 8.76% PyTorch Flash BF16; for 9B, 9.35%, 9.27% and 9.28% respectively.
Even the two PyTorch BF16 attention implementations differed at cosine
0.998646 (4B) and 0.998455 (9B). This is evidence of numerical sensitivity,
not proof of full vision compatibility: the original native-vs-reference
gate is still failing and needs separate resolution before small-model
multimodal experiments.
