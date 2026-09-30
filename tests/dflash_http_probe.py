#!/usr/bin/env python3
"""Live QW3 completion regression; requires a separately launched GPU server."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = args.base_url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    text = "<|im_start|>user\nCompute 17 * 23 and explain the calculation.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    results = []

    def request(body, disconnect=False):
        start = time.perf_counter()
        req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            if not body.get("stream"):
                raw = json.load(response)
                result = dict(text=raw["choices"][0]["text"], usage=raw["usage"], response=raw)
            else:
                chunks, first_text, done, usage = [], None, False, None
                for wire in response:
                    if not wire.startswith(b"data: "):
                        continue
                    payload = wire[6:].strip()
                    if payload == b"[DONE]":
                        done = True
                        break
                    chunk = json.loads(payload)
                    chunks.append(chunk)
                    delta = "".join(c.get("text", "") for c in chunk.get("choices", []))
                    if delta and first_text is None:
                        first_text = time.perf_counter() - start
                        if disconnect:
                            break
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                if not disconnect:
                    assert done and usage, "missing stream terminator or usage"
                result = dict(text="".join(c.get("text", "") for x in chunks for c in x.get("choices", [])),
                    usage=usage, chunks=chunks, ttft_s=first_text, disconnected=disconnect)
        result["wall_s"] = time.perf_counter() - start
        return result

    for order in ("joint", "top-k-first"):
        body = dict(model=model, prompt=ids, max_tokens=32, temperature=1., top_p=.95,
            top_k=20, sampling_filter_order=order, seed=101, ignore_eos=True)
        first = request(body)
        assert first["usage"]["completion_tokens"] == 32
        assert first["usage"]["prompt_tokens"] == len(ids)
        results.append(dict(case=order + ":tokens", result=first))
        repeated = request(body)
        assert repeated["text"] == first["text"], "seeded repeat changed"
        results.append(dict(case=order + ":repeat", result=repeated))
        rendered = request(dict(body, prompt=text))
        assert rendered["text"] == first["text"], "raw text and raw token prompt differ"
        results.append(dict(case=order + ":text", result=rendered))
        streaming = dict(body, stream=True, stream_options={"include_usage": True})
        streamed = request(streaming)
        assert streamed["text"] == first["text"], "streaming changed output"
        assert streamed["usage"]["completion_tokens"] == 32
        results.append(dict(case=order + ":stream", result=streamed))
        stop = first["text"][3:18]
        assert stop
        stopped = request(dict(streaming, stop=stop))
        assert stopped["text"] == first["text"].split(stop)[0], "stop bytes leaked from stream"
        assert stopped["usage"]["completion_tokens"] < 32, "stop did not halt decoding"
        assert any(c.get("finish_reason") == "stop" for x in stopped["chunks"] for c in x.get("choices", []))
        results.append(dict(case=order + ":stop", result=stopped))
        chinese = dict(body, prompt="<|im_start|>user\n用中文解释为什么天空是蓝色的。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        chinese_reference = request(chinese)
        assert any(ord(c) > 127 for c in chinese_reference["text"]), "UTF-8 case generated only ASCII"
        chinese_stream = request(dict(chinese, stream=True, stream_options={"include_usage": True}))
        assert chinese_reference["text"] == chinese_stream["text"], "UTF-8 stream changed output"
        results.append(dict(case=order + ":utf8", reference=chinese_reference, result=chinese_stream))
        results.append(dict(case=order + ":disconnect", requested_tokens=512,
            result=request(dict(streaming, max_tokens=512), disconnect=True)))
        recovered = request(body)
        assert recovered["text"] == first["text"], "disconnect recovery changed output"
        results.append(dict(case=order + ":after-disconnect", result=recovered))
    try:
        request(dict(body, sampling_filter_order="invalid"))
        raise AssertionError("invalid filter order accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 400, f"invalid filter order returned {error.code}"
        results.append(dict(case="invalid-policy", status=error.code, response=json.loads(error.read())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="pass", cases=len(results), base_url=base,
        tokenizer=args.tokenizer, prompt_ids=ids, results=results), indent=2) + "\n")
    print(f"QW3 HTTP checks={len(results)} status=pass output={args.output}")


if __name__ == "__main__":
    main()
