#!/usr/bin/env python3
"""Live KVMem NFC span regression, not a task-quality or throughput benchmark."""
import argparse
import copy
import json
from pathlib import Path
import unicodedata
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = args.base_url.rstrip("/")
    with opener.open(base + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    record = {"status": "running", "performance_qualified": False, "results": []}

    def save():
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    def request(label, body, harness=False):
        headers = {"Content-Type": "application/json"}
        if harness:
            headers["x-qw3-harness"] = "mini-swe-agent"
        req = urllib.request.Request(base + "/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(), headers=headers)
        try:
            with opener.open(req, timeout=180) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{label}: HTTP {error.code}: {error.read().decode()}") from error
        assert result["usage"]["completion_tokens"] == 8, (label, result)
        record["results"].append({"case": label, "request": body, "harness": harness, "response": result})
        save()
        return result

    def body_for(kind, form):
        normalize = lambda text: unicodedata.normalize(form, text)
        history = normalize("The café stores Ångström readings beside 각. " +
                            "An ordinary archived log line. " * 80 +
                            "The current retry limit is three.")
        messages = [
            {"role": "system", "content": normalize("Read the café logs and answer briefly.")},
            {"role": "user", "content": history},
            {"role": "assistant", "content": normalize("I saved the café readings.")},
            {"role": "user", "content": normalize("What is the café retry limit? Reply briefly.")},
        ]
        body = {"model": model, "messages": messages, "temperature": 0, "seed": 101,
                "max_tokens": 8, "ignore_eos": True, "enable_thinking": False}
        if kind == "explicit":
            body["kvmem_retrieval_group_spans"] = [
                {"message_index": 1, "content_start": 0, "content_end": len(history.encode())}]
        if kind == "tool_suffix":
            body["tools"] = [{"type": "function", "function": {
                "name": "read_log", "description": "Read a log.",
                "parameters": {"type": "object", "properties": {}}}}]
            messages.extend([
                {"role": "assistant", "content": normalize("Checking the café log."),
                 "tool_calls": [{"id": "call_1", "type": "function", "function": {
                     "name": "read_log", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": normalize(
                    "The café retry limit is three; the Ångström sensor is healthy.")},
            ])
        return body

    try:
        for kind in ("automatic", "explicit", "harness", "tool_suffix"):
            results = [request(kind + ":" + form, body_for(kind, form),
                               kind in ("harness", "tool_suffix")) for form in ("NFC", "NFD")]
            assert results[0]["choices"][0]["message"] == results[1]["choices"][0]["message"], kind
            for field in ("prompt_tokens", "completion_tokens"):
                assert results[0]["usage"][field] == results[1]["usage"][field], (kind, field)
        # Two raw disjoint spans share one composed grapheme after normalization.
        # Both must remain valid retrieval groups after outward token rounding.
        split_body = body_for("explicit", "NFD")
        raw = split_body["messages"][1]["content"].encode()
        begin = raw.index("e\u0301".encode())
        split_body["kvmem_retrieval_group_spans"] = [
            {"message_index": 1, "content_start": begin, "content_end": begin + 1},
            {"message_index": 1, "content_start": begin + 1, "content_end": begin + 3}]
        request("split_composed_grapheme", split_body)
        record.update(status="passed", response_count=len(record["results"]), nfc_nfd_pairs=4)
        save()
        print(f"KVMem Unicode HTTP responses={len(record['results'])} paired_cases=4 PASS")
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
