# Qwen Q8 MTP for external retrieval methods

Goal: implement and measure real Qwen3.8-27B MTP for the isolated InfLLM and
InfiniGen ports, without modifying production qw3 or previous frozen results.

* Exact deployed Q8_0 GGUF, FP8 historical KV, greedy decoding, batch one,
  four draft tokens. The existing standalone Q8 projection library is reused;
  it contains frozen qw3 arithmetic kernels. Retrieval is the authors' code,
  adapted to Qwen's hybrid architecture; this is not an untouched upstream run.
* MTP loads the actual next-token predictor in GGUF block 64, sharing the main
  embedding and output head. Priming pairs token t with pre-final-norm target
  hidden state t-1 (zero at t=0). Rejected draft KV is overwritten from accepted
  target hidden states. Draft attention uses an explicitly reported full FP8
  GPU history; target retrieval remains method-specific.
* Verification batches projections and MLPs but invokes retrieval and recurrent
  state transitions once per token, in causal order. Never average speculative
  queries, freeze a retrieval set, or crop a recurrent cache by length alone.
* Retain bounded prefix snapshots within the draft batch, including InfLLM
  local scores/block representatives and InfiniGen partial-index lengths.
  Validate rejection at different offsets, block crossing, subsequent decode,
  logits/selected IDs/state consistency against independent serial prefills.
* Measure MTP=0 and MTP=4 with the same external backend; separate draft,
  verification, state maintenance, accepted tokens and total decode time.
  Reuse the frozen native Q8/MTP report as an explicitly different-backend
  deployment reference. Do not claim algorithm-only speedup or task accuracy.
* Start with short correctness and throughput probes; then run identical
  8K/32K/120K/240K inputs, 128 outputs, three repetitions where validated.
  Retain errors, failed checks and slow results. No 50 token/s acceptance target.
* OSWorld continues during CPU implementation. Before GPU work, archive the
  unfinished arm and pause its owned processes/VM. Preserve completed results;
  restore the frozen suite and verify successful API traffic after testing.

Sources: pinned InfLLM 12b70798f56e56ebb23c53c7018091a3f540a028;
InfiniGen 3754f52d131c30ab7a310e3c4cf356e8e3326ac3; local Qwen MTP math;
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py
