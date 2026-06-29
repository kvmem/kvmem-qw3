#!/usr/bin/env python3
"""DeepSeek answer-equivalence judge for the KVMem LongMemEval evaluation.

Mirrors the motivation study (docs/motivation_experiment_summary_en.md §3.2): a
strong LLM judges whether the model's answer is equivalent to the gold answer.
The judge prompts are the *official* LongMemEval per-type templates
(`get_anscheck_prompt` from the LongMemEval repo), which apply type-specific
leniency:

  - single-session-* / multi-session : answer must contain the correct answer
  - temporal-reasoning               : tolerate off-by-one day/week/month errors
  - knowledge-update                 : correct if the *updated* answer is present
  - single-session-preference        : gold is a rubric; correct if personalized
  - abstention (_abs)                : correct if the model declines to answer

API key comes strictly from the environment (`DEEPSEEK_API_KEY`); it is never
hardcoded or written to disk. Base URL and model are env-overridable so the same
harness can target whichever DeepSeek tier the user has access to.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

try:
    from .dataset import Sample, is_abstention
except ImportError:  # allow running as a loose module
    from dataset import Sample, is_abstention  # type: ignore


def get_anscheck_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """Official LongMemEval judge prompt, selected by question type."""
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that the "
            "information is incomplete, or some other information is given but the "
            "asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel "
            "Response: {}\n\nDoes the model correctly identify the question as "
            "unanswerable? Answer yes or no only."
        ).format(question, answer, response)

    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no. If the response is equivalent to the correct "
            "answer or contains all the intermediate steps to get the correct "
            "answer, you should also answer yes. If the response only contains a "
            "subset of the information required by the answer, answer no. \n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the "
            "model response correct? Answer yes or no only."
        )
    elif task == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no. If the response is equivalent to the correct "
            "answer or contains all the intermediate steps to get the correct "
            "answer, you should also answer yes. If the response only contains a "
            "subset of the information required by the answer, answer no. In "
            "addition, do not penalize off-by-one errors for the number of days. If "
            "the question asks for the number of days/weeks/months, etc., and the "
            "model makes off-by-one errors (e.g., predicting 19 days when the "
            "answer is 18 days), the model's response is still correct. \n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the "
            "model response correct? Answer yes or no only."
        )
    elif task == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no. If the response contains some previous "
            "information along with an updated answer, the response should be "
            "considered as correct as long as the updated answer is the required "
            "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. The "
            "model does not need to reflect all the points in the rubric. The "
            "response is correct as long as it recalls and utilizes the user's "
            "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel "
            "Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    else:
        # Unknown type: fall back to the generic containment prompt.
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
            "Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    return template.format(question, answer, response)


@dataclass
class JudgeResult:
    correct: bool
    raw: str                  # judge's raw text (e.g. "yes" / "no")
    error: str | None = None


class DeepSeekJudge:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 4,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set; the judge needs a DeepSeek API key "
                "(supply via the DEEPSEEK_API_KEY environment variable)."
            )
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ).rstrip("/")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()

    def _call(self, prompt: str) -> tuple[str, str | None]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }
        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    msg = data["choices"][0]["message"]
                    return (msg.get("content") or "").strip(), None
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                # 429 / 5xx: back off and retry; 4xx (except 429): give up.
                if r.status_code not in (429,) and 400 <= r.status_code < 500:
                    break
            except requests.RequestException as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 16))
        return "", last_err

    def judge(self, sample: Sample, response: str) -> JudgeResult:
        abst = is_abstention(sample)
        prompt = get_anscheck_prompt(
            sample.question_type, sample.question, sample.answer, response, abstention=abst
        )
        raw, err = self._call(prompt)
        if err is not None:
            return JudgeResult(correct=False, raw=raw, error=err)
        verdict = raw.strip().lower()
        # The judge is instructed to answer yes/no only; match the leading token.
        correct = verdict.startswith("yes") or verdict == "y"
        return JudgeResult(correct=correct, raw=raw)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    try:
        from .dataset import build_subset, load_all
    except ImportError:
        from dataset import build_subset, load_all  # type: ignore

    ap = argparse.ArgumentParser(description="Smoke-test the DeepSeek judge on a fabricated answer.")
    ap.add_argument("--data", type=Path,
                    default=Path("/data/chaidi/kvmem_eval/data/longmemeval_s.json"))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--response", default=None,
                    help="model response to grade; defaults to the gold answer (should judge 'yes')")
    args = ap.parse_args()

    s = build_subset(load_all(args.data))[args.index]
    resp = args.response if args.response is not None else s.answer
    j = DeepSeekJudge()
    print(f"[{s.question_id}] type={s.question_type}")
    print(f"  question: {s.question}")
    print(f"  gold:     {s.answer}")
    print(f"  response: {resp}")
    r = j.judge(s, resp)
    print(f"  -> correct={r.correct} raw={r.raw!r} error={r.error}")
