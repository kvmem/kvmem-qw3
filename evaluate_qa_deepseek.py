from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    # Copied from the official LongMemEval evaluate_qa.py prompt templates.
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError(task)
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


def load_deepseek_config(path: str | None) -> tuple[str, str]:
    """Resolve (api_key, model). Prefer the DEEPSEEK_API_KEY environment variable
    so the key is never written to disk or committed; fall back to the config
    file only when the env var is unset. DEEPSEEK_MODEL overrides the model."""
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key, os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").lower()
    if not path:
        raise ValueError(
            "no DeepSeek key: set DEEPSEEK_API_KEY env var or pass --config"
        )
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty DeepSeek config: {path}")
    api_key = lines[0]
    if ":" in api_key:
        provider, key = api_key.split(":", 1)
        if provider.lower() in {"deepseek", "ds", "dsv4pro"}:
            api_key = key
    model = lines[1].lower() if len(lines) > 1 else "deepseek-v4-pro"
    return api_key, model


def load_json_records(text: str) -> list:
    """Parse a list of records from JSON array, JSONL, or concatenated/pretty-printed
    JSON objects. The provided selected_12_samples.jsonl is a stream of concatenated
    (pretty-printed) JSON objects, so fall back to a raw_decode loop that tolerates
    arbitrary whitespace (incl. newlines) between objects. Mirrors dataset.py load_all."""
    try:
        raw = json.loads(text)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return [raw]
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    raw = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        raw.append(obj)
        idx = end
    return raw


def call_chat(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    retries: int = 5,
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            text = raw["choices"][0]["message"].get("content") or ""
            return text.strip(), raw
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp-file", required=True)
    parser.add_argument("--ref-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None,
                        help="DeepSeek config file (ignored when DEEPSEEK_API_KEY env is set)")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    api_key, model = load_deepseek_config(args.config)
    hypotheses = [json.loads(line) for line in Path(args.hyp_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        hypotheses = hypotheses[: args.limit]
    references = load_json_records(Path(args.ref_file).read_text(encoding="utf-8"))
    qid2ref = {entry["question_id"]: entry for entry in references}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    with out.open("w", encoding="utf-8") as f:
        for idx, entry in enumerate(hypotheses, start=1):
            qid = entry["question_id"]
            ref = qid2ref[qid]
            prompt = get_anscheck_prompt(
                ref["question_type"],
                ref["question"],
                ref["answer"],
                entry["hypothesis"],
                abstention="_abs" in qid,
            )
            response, raw = call_chat(api_key, args.base_url, model, prompt, args.max_tokens)
            label = "yes" in response.lower()
            entry = {
                **entry,
                "autoeval_label": {
                    "model": model,
                    "label": label,
                    "response": response,
                },
                "judge_raw_usage": raw.get("usage", {}),
            }
            logs.append(entry)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(hypotheses)}] {qid} judge={response!r} label={label}", flush=True)
    acc = sum(1 for row in logs if row["autoeval_label"]["label"]) / max(len(logs), 1)
    print(f"Accuracy: {acc:.4f} ({sum(1 for row in logs if row['autoeval_label']['label'])}/{len(logs)})")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
