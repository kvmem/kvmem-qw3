# Whole-history summary evaluation

`QW3_OSWORLD_HISTORY_MODE=summary` changes the Qwen agent's history policy.
Between summary events, every screenshot and assistant response is appended to an
unchanged transcript. Both the `history_n` sliding window and periodic image
folding are disabled. Their old numerical settings remain recorded but inactive.

Before an action request approaches `QW3_OSWORLD_SUMMARY_TRIGGER_TOKENS` (default
196608), the same local model summarizes all completed old screenshots and text,
including the previous summary. The new screenshot is excluded from the summary
source and retained for the next action. A fixed system protocol allows a tagged
tool response to request a summary instead of the normal action format, preserving
the prefix even for the summary-generation call; no summary is executed as
a desktop action. The replacement contains the unchanged original system/task
instructions, the summary and the current screenshot. Old images and responses
remain in the audit trail but are no longer submitted to the model.

The budget estimate is anchored to the preceding action's actual server prompt
usage. Newly serialized response text is counted using `/v1/tokenize`, and new
Qwen image tokens are counted from the processed 32-pixel patch-merge grid. A
2048-token margin covers wrappers/rendering differences. The first request and
post-summary request count their full text. This is a conservative estimate, not
an exact server-side chat-template count. The trigger must leave room for the
larger of the normal output and summary budgets plus the margin.

The default summary output cap is 4096 tokens. Summary calls read natural-language
content rather than replaying raw action XML. Tool/reasoning response stop markers
and defensive suffix removal retain usable summary prose while discarding appended
commands. A native `tool_calls` finish reason alone does not invalidate usable text;
tool-only, too-short, reasoning-only, and truncated outputs remain invalid. Summary
outputs are never passed to GUI action parsing or execution.

Invalid summaries receive up to three attempts with shorter requested lengths. The
first two preserve the old prefix; the last uses a dedicated summarizer system
prompt containing all old images/text as historical data and may need a full prefill.
History changes only after validation succeeds. If all attempts fail, the old
history remains intact. Each attempt's source and returned text are archived in
`history_summaries/`; attempt metadata (including raw completion and tool calls),
the final event and total compaction latency appear in `history_compaction.jsonl`
and the containing step. Transport errors retain the
existing cancellation-ack retry protocol.

Example settings for task 003:

```bash
QW3_OSWORLD_HISTORY_MODE=summary
QW3_OSWORLD_CONTEXT_TOKENS=262144
QW3_OSWORLD_SUMMARY_TRIGGER_TOKENS=196608
QW3_OSWORLD_SUMMARY_MAX_TOKENS=4096
QW3_OSWORLD_ENABLE_THINKING=0
QW3_OSWORLD_PRESERVE_THINKING=0
QW3_OSWORLD_THINKING_BUDGET=0
QW3_OSWORLD_MAX_OUTPUT_TOKENS=16384
QW3_OSWORLD_PREFIX_CACHE=1
QW3_OSWORLD_MTP_CHAIN=4
QW3_OSWORLD_KVMEM_PREFIX_CACHE=0
QW3_VISION_GPU_CACHE_MIB=4096
```

The 4 GiB vision cache is a capacity ceiling and grows on demand. It avoids
re-encoding the long screenshot history before a summary. Logical context and
physical main/MTP KV pools remain 262144 slots each.

This is a new agent configuration: scores must be labeled with this policy and
thinking disabled. A summary can omit useful facts; the saved original task and
fresh screenshot do not by themselves guarantee unchanged task accuracy.

Validation: `test_summary_history.py` covers append stability beyond the legacy
limits, repeated replacement, transaction failure and output reservation. The
real-model `summary_history_canary.py` forces two summaries at a smaller trigger
while retaining a 256K model context and MTP=4, then verifies prefix reuse on the
following ordinary action turns. It does not execute generated GUI actions.

`replay_summary_failures.py --source-run <run> --out <directory>` restores the
saved failing histories for tasks 005, 007, 008, 010 and 012. It starts its own
256K/MTP=4 server on port 18081, replays each original long prompt, checks summary
prefix reuse above 95%, replaces the old history, and verifies a prefix hit on the
following action turn. It archives per-case attempts and a final `report.json`.
This is a format/cache regression check; generated actions are not executed and
it does not produce benchmark scores. Release the GPU before running it.
