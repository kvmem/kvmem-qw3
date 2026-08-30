# KVMem public demo storyboard

Status: planning. Do not publish performance or capacity claims from this plan;
record them from the final release profile and attach the resulting manifest.

## Recommendation

Use two demonstrations with different jobs:

1. **Launch proof:** a generated incident ledger with an unseen code. This is
   the primary evidence because the model cannot have memorized the answer.
2. **Story showcase:** four fixed public-domain Chinese-classic excerpts. This
   is more memorable, but should follow the proof rather than replace it.

The launch proof should be the video embedded near the top of the README.

## Launch proof: the long incident ledger

### Claim being demonstrated

QW3 can ingest an accumulated history larger than its active KVMem budget,
move cold KV blocks out of the GPU working set, use query-conditioned `mean-k`
without fallback, and recover a fact placed outside mandatory sink/recent
regions.

It does **not** by itself prove lossless retrieval, a general accuracy rate,
1M-token support, 24 GiB support, or a specific VRAM saving.

### Reproducible input

- Generate 6,000 deterministic incident records during the recording.
- Insert one randomly generated rollback code between records 2,999 and 3,000.
- Use a fixed corpus seed, but generate the code after the model release so it
  cannot occur in pretraining.
- Record the corpus SHA-256, code SHA-256, QW3 commit, model revision/SHA-256,
  tokenizer count, and full command in a machine-readable manifest.
- Publish the exact corpus fixture after recording.

Canonical question:

```text
Reply with only the exact rollback code recorded in the middle of the incident ledger.
```

The current README fixture is 106,908 raw-history tokens with the Qwen3.6
tokenizer. The canonical profile uses `ctx=131072`, `kvmem-budget=32768`, and
128-token blocks.

### Recording layout

Use one uncluttered 16:9 screen with three persistent regions:

- **Left:** corpus length, target position, model/commit, and four configuration
  values (`ctx`, budget, block size, tier sizes).
- **Center:** an animated block timeline colored by GPU, CPU, and NVMe tier;
  highlight selected blocks only after retrieval completes.
- **Right:** the user question, generated answer, and exact-match result.

Show only filtered trace events, not a scrolling wall of logs:

- `[kvmem-scorer] ... requested=mean-k used=mean-k fallback=0`
- `native kvmem query-replay (plain) ... index_ready=1`
- `[kvmem-tier] stage_out ...`
- `[kvmem-tier] stage_in ...`

### 60–75 second storyboard

| Time | Screen action |
|---|---|
| 0–6 s | Hook: “106,908-token history · 32,768-token GPU window · one unseen code.” |
| 6–15 s | Generate the corpus; show seed, corpus hash, target position, and hidden code hash. |
| 15–24 s | Show the public Q8 model SHA, QW3 commit, GPU, and canonical KVMem settings. |
| 24–39 s | Submit the question. If prefill is shortened in editing, label the segment `time-lapse` and retain the real wall time. |
| 39–53 s | Animate tier movement and show the four filtered trace checks. |
| 53–63 s | Reveal answer and ground truth side-by-side; display `EXACT MATCH`. |
| 63–75 s | Show the evidence manifest and a concise boundary: “single visible case; sparse selection is lossy; multi-seed results linked.” |

### Publication gate

Do not publish the video until all of these pass:

- Final public QW3 commit and public Q8 revision/SHA are used.
- Rendered prompt is below `ctx`; raw history is above the KVMem budget.
- The target is outside sink and recent mandatory regions.
- `mean-k` is used with `fallback=0` and `index_ready=1`.
- At least one CPU `stage_out` and one subsequent `stage_in` occur.
- The answer exactly matches the generated code.
- Five clean-process reruns pass five out of five.
- A deletion counterfactual does not reproduce the code.
- A replacement counterfactual returns the replacement code.
- Published telemetry separates GPU KV, total process VRAM, CPU tier, and NVMe;
  token/budget ratio is never presented as VRAM savings.

If the trace cannot map the target block itself from stage-out to stage-in, say
only that KVMem and tiering engaged. Do not claim that the specific target block
was reloaded until that mapping is recorded.

### Release artifacts

Publish alongside the video:

- `manifest.json`: code/model/toolchain/hardware/config/checksums.
- `corpus.txt` or a deterministic corpus generator plus its output hash.
- `request.json` and `response.json`.
- Filtered trace and raw log with local paths and host identifiers removed.
- Aggregated block-position/tier data with token IDs removed.
- Peak VRAM/RSS samples and unedited timing fields.
- Multi-seed and counterfactual result summary.

## Story showcase: four Chinese classics excerpts

The complete Four Great Classical Novels are a poor launch proof:

- common questions may be answered from model pretraining;
- complete editions are likely far beyond the current 128K public profile;
- edition, punctuation, and simplified/traditional differences change tokens;
- a modern annotated edition may carry separate rights.

A safe and honest second video can use **four excerpts**, totaling roughly
80K–110K tokens:

- Pin a public-domain or compatibly licensed source revision for each excerpt.
- Record source URLs, revisions, attribution, license, and SHA-256.
- Insert one demo-only editorial note with a newly generated code into the
  middle of each excerpt.
- Ask one natural passage question and one code canary per excerpt, in shuffled
  order.
- End with one cross-excerpt question that requires selecting blocks from two
  works.
- Title the video “Four Chinese classics excerpts in one KVMem session,” never
  “the complete Four Great Classical Novels.”

This version keeps the cultural hook while the generated notes prove that the
answers came from the supplied context rather than memorized literary facts.

## README media plan

- Keep the lightweight flow graphic in `docs/assets/kvmem-flow.svg`.
- Add a static poster only after a real release-profile recording exists.
- Upload the final H.264 MP4 through GitHub and replace `Demo video: TBD` with
  its GitHub-hosted URL; do not commit a large changing video into Git history.
- Keep the video concise, provide captions or a transcript if it has narration,
  and never let animation be the only explanation of the flow.
