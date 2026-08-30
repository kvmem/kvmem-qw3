"""Runtime agent used by the dense-256K MiniSWE compaction baseline.

This module is copied into the pinned mini-swe-agent tool environment by
``CompactingMiniSweAgent``.  It deliberately leaves the official prompt,
tools, action parser, and execution loop unchanged.  The only additional
behavior is a context-window guard that asks the same model for a durable work
summary and replaces the old message transcript before the next solver call.
"""

from __future__ import annotations

import math
import time
from typing import Any

import litellm

from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from minisweagent.models.utils.content_string import get_content_string


class CompactingInteractiveAgentConfig(InteractiveAgentConfig):
    """Configuration for a 256K dense-context compaction baseline."""

    compaction_trigger_tokens: int = 220_000
    """Compact before the estimated next prompt reaches this size."""

    compaction_summary_max_tokens: int = 16_384
    """Maximum generated tokens for a single compact summary."""

    compaction_retry_attempts: int = 3
    """Transient summary-generation retries before failing the task."""


_SUMMARY_INSTRUCTION = """\
PRIVATE CONTEXT COMPACTION TASK

Compress the complete software-engineering trajectory above into one durable,
standalone continuation record.  Preserve the original task and every detail
needed to continue correctly: exact file paths and symbols, repository state,
implemented edits, commands and important outputs, failing and passing tests,
errors, hypotheses already ruled out, constraints, and unfinished next steps.
Do not solve the task, call tools, emit a tool call, or claim work was completed
unless the transcript proves it.  Prefer concrete facts over narrative.  The
next solver will receive the original task plus only this record, so omit no
state that could affect correctness.  Output only the continuation record.
"""


class CompactingInteractiveAgent(InteractiveAgent):
    """The official interactive agent with bounded automatic compaction."""

    def __init__(
        self,
        *args: Any,
        config_class: type = CompactingInteractiveAgentConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, config_class=config_class, **kwargs)
        self.compaction_records: list[dict[str, Any]] = []

    @staticmethod
    def _message_chars(message: dict[str, Any]) -> int:
        try:
            return len(get_content_string(message))
        except Exception:
            return len(str(message.get("content", "")))

    def _estimated_next_prompt_tokens(self) -> int:
        """Estimate the next prompt from the last exact server usage.

        QW3 returns exact prompt/completion token counts on every successful
        solver response.  Only the observation messages added after that
        response need estimating.  The official MiniSWE observation template
        caps command output at 10K characters, so a conservative two
        characters/token conversion plus framing headroom is sufficient to
        trigger well before the hard 256K boundary.
        """

        for index in range(len(self.messages) - 1, -1, -1):
            message = self.messages[index]
            # A normal response stores usage on the assistant message.  A
            # MiniSWE FormatError stores the rejected response and its usage on
            # the injected user correction instead, so inspect every role.
            response = message.get("extra", {}).get("response", {})
            usage = response.get("usage") if isinstance(response, dict) else None
            if not isinstance(usage, dict):
                continue
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens", 0)
            if not isinstance(prompt, int) or prompt < 0:
                continue
            if not isinstance(completion, int) or completion < 0:
                completion = 0
            suffix_chars = sum(
                self._message_chars(item) for item in self.messages[index + 1 :]
            )
            suffix_tokens = math.ceil(suffix_chars / 2.0)
            return prompt + completion + suffix_tokens + 1024

        total_chars = sum(self._message_chars(item) for item in self.messages)
        return math.ceil(total_chars / 2.0) + 2048

    @staticmethod
    def _is_context_window_error(exc: BaseException) -> bool:
        if isinstance(exc, litellm.exceptions.ContextWindowExceededError):
            return True
        text = f"{type(exc).__name__}: {exc}".lower()
        return (
            "context window" in text
            or "maximum context" in text
            or "prompt exceeds" in text
            or "context_length_exceeded" in text
            or "status code: 413" in text
        )

    def _generate_summary(self) -> tuple[str, dict[str, Any]]:
        prepare = getattr(self.model, "_prepare_messages_for_api", None)
        if not callable(prepare):
            raise RuntimeError("compaction requires a LiteLLM-compatible model")
        messages = list(prepare(self.messages))
        messages.append({"role": "user", "content": _SUMMARY_INSTRUCTION})

        config = getattr(self.model, "config", None)
        model_name = getattr(config, "model_name", "")
        if not model_name:
            raise RuntimeError("compaction model has no configured model_name")
        kwargs = dict(getattr(config, "model_kwargs", {}) or {})
        extra_body = dict(kwargs.get("extra_body", {}) or {})
        extra_body["enable_thinking"] = False
        kwargs.update(
            {
                "tools": [BASH_TOOL],
                "tool_choice": "none",
                "max_tokens": self.config.compaction_summary_max_tokens,
                "temperature": 0.0,
                "extra_body": extra_body,
            }
        )

        last_error: BaseException | None = None
        for attempt in range(1, self.config.compaction_retry_attempts + 1):
            try:
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                    **kwargs,
                )
                choice = response.choices[0]
                message = choice.message
                message_dict = (
                    message.model_dump(mode="json")
                    if hasattr(message, "model_dump")
                    else dict(message)
                )
                summary = get_content_string(message_dict).strip()
                if not summary:
                    raise RuntimeError("compaction model returned an empty summary")
                usage = getattr(response, "usage", None)
                usage_dict = (
                    usage.model_dump(mode="json")
                    if hasattr(usage, "model_dump")
                    else dict(usage or {})
                )
                return summary, {
                    "attempt": attempt,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "usage": usage_dict,
                }
            except BaseException as exc:
                last_error = exc
                if attempt >= self.config.compaction_retry_attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"compaction failed: {last_error}")

    def _compact_history(self, reason: str) -> None:
        if len(self.messages) < 3:
            raise RuntimeError("cannot compact a trajectory without history")
        system = next(
            (message for message in self.messages if message.get("role") == "system"),
            None,
        )
        root = next(
            (message for message in self.messages if message.get("role") == "user"),
            None,
        )
        if system is None or root is None:
            raise RuntimeError("compaction requires the original system and user messages")

        estimated_before = self._estimated_next_prompt_tokens()
        messages_before = len(self.messages)
        started = time.monotonic()
        summary, response_meta = self._generate_summary()
        elapsed = time.monotonic() - started

        # Keep the official system prompt and original task verbatim.  The
        # generated continuation record replaces all intermediate turns; a
        # final user instruction makes the next ordinary MiniSWE step causal.
        self.messages = [
            dict(system),
            dict(root),
            {
                "role": "assistant",
                "content": "CONTEXT COMPACTION SUMMARY\n\n" + summary,
            },
            {
                "role": "user",
                "content": (
                    "Continue solving the original task from the compacted "
                    "state above. Verify the current repository state before "
                    "making further changes, and use the normal bash tool."
                ),
            },
        ]
        record = {
            "index": len(self.compaction_records) + 1,
            "reason": reason,
            "estimated_prompt_tokens_before": estimated_before,
            "messages_before": messages_before,
            "messages_after": len(self.messages),
            "summary_chars": len(summary),
            "elapsed_seconds": elapsed,
            **response_meta,
        }
        self.compaction_records.append(record)
        print(
            "[qw3-compaction] "
            f"index={record['index']} reason={reason} "
            f"estimated_prompt_tokens={estimated_before} "
            f"messages={messages_before}->{len(self.messages)} "
            f"summary_chars={len(summary)} elapsed_s={elapsed:.3f}",
            flush=True,
        )

    def query(self) -> dict:
        estimated = self._estimated_next_prompt_tokens()
        if estimated >= self.config.compaction_trigger_tokens:
            self._compact_history("threshold")
        try:
            return super().query()
        except BaseException as exc:
            if not self._is_context_window_error(exc):
                raise
            # The threshold includes ample headroom, but retain a hard-window
            # recovery for unusually large protocol/tool framing.  Compact the
            # still-intact transcript and retry exactly once.
            self._compact_history("context-window-recovery")
            return super().query()

    def serialize(self, *extra_dicts: dict[str, Any]) -> dict:
        compaction = {
            "info": {
                "compaction": {
                    "enabled": True,
                    "trigger_tokens": self.config.compaction_trigger_tokens,
                    "summary_max_tokens": self.config.compaction_summary_max_tokens,
                    "count": len(self.compaction_records),
                    "records": self.compaction_records,
                }
            }
        }
        return super().serialize(*extra_dicts, compaction)
